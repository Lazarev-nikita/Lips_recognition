import os
import random
import pathlib
import collections
import tensorflow as tf
# import keras
import numpy as np
# from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import einops

print(tf.test.gpu_device_name())

class FileOps:

    @staticmethod
    def get_files_list(src_dir: str):
        files_list = os.listdir(src_dir)
        return files_list

    @staticmethod
    def get_files_per_class(files):
        files_per_class = collections.defaultdict(list)
        for fname in files:
            class_name = FileOps.get_class(fname)
            files_per_class[class_name].append(fname)
        return files_per_class

    @staticmethod
    def get_class(fname):
        return fname.split('_')[0]

    @staticmethod
    def split_class_lists(files_per_class, count):
        split_files = []
        remainder = {}
        for cls in files_per_class:
            split_files.extend(files_per_class[cls][:count])
            remainder[cls] = files_per_class[cls][count:]
        return split_files, remainder


class FrameGenerator:
    def __init__(self, src_path, files, n_frames, training=False):
        self.path = src_path
        self.files = files
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(FileOps.get_class(p) for p in files))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        paths = list(self.path / f for f in self.files)
        classes = list(FileOps.get_class(f) for f in self.files)
        return paths, classes

    @staticmethod
    def frames_from_video_file(video_path, n_frames, output_size=(256, 256)):
        # Read each video frame by frame
        result = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return result

        for i in range(n_frames):
            ret, frame = cap.read()
            if ret:
                result.append(FrameGenerator.format_frames(frame, output_size))
            else:
                if len(result) == 0:
                    print(f'file={video_path}; frame = {i}; len(result)={len(result)}')
                result.append(np.zeros_like(result[0]))
        cap.release()

        result = np.array(result)[..., [2, 1, 0]]

        return result

    @staticmethod
    def get_frames_count(video_path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return 0
        frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return int(frames_count)


    @staticmethod
    def format_frames(frame, output_size):
        resized_image = cv2.resize(frame, output_size, interpolation = cv2.INTER_CUBIC)
        resized_image = resized_image.astype(np.float32)
        resized_image /= 255.
        return resized_image

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = FrameGenerator.frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)

    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label = 'train')
    ax1.plot(history.history['val_loss'], label = 'test')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    max_loss = max(history.history['loss'] + history.history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'],  label = 'train')
    ax2.plot(history.history['val_accuracy'], label = 'test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.show()

def get_actual_predicted_labels(dataset):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return actual, predicted


def plot_confusion_matrix(actual, predicted, labels, ds_type):
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize':(12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

def calculate_classification_metrics(y_actual, y_pred, labels):
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    tp = np.diag(cm) # Diagonal represents true positives
    precision = dict()
    recall = dict()
    for i in range(len(labels)):
        col = cm[:, i]
        fp = np.sum(col) - tp[i] # Sum of column minus true positive is false negative

        row = cm[i, :]
        fn = np.sum(row) - tp[i] # Sum of row minus true positive, is false negative

        precision[labels[i]] = tp[i] / (tp[i] + fp) # Precision

        recall[labels[i]] = tp[i] / (tp[i] + fn) # Recall

    return precision, recall


class ResidualMain(tf.keras.layers.Layer):
    """
      Residual block of the model with convolution, layer normalization, and the
      activation function, ReLU.
    """
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class Project(tf.keras.layers.Layer):
    """
      Project certain dimensions of the tensor as the data is passed through different
      sized filters and downsampled.
    """
    def __init__(self, units):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

def add_residual_block(input, filters, kernel_size):
    """
      Add residual blocks to the model. If the last dimensions of the input data
      and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters,
                       kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return tf.keras.layers.add([res, out])


# SOURCE_FOLDER = '/content/drive/My Drive/lips_data'
SOURCE_FOLDER = 'C:\\Projects\\my\\python\\lips_data'
SOURCE_VIDEO_FOLDER = SOURCE_FOLDER + '/crop-video'
HEIGHT = 256
WIDTH = 256
original_data_dir = pathlib.Path(SOURCE_VIDEO_FOLDER)
video_files = FileOps.get_files_list(SOURCE_VIDEO_FOLDER)
files_for_class = FileOps.get_files_per_class(video_files)

classes = list(files_for_class.keys())
print(f'Video classes count: {len(classes)}')

#random files for each class
print('Shuffle video for each class')
for cls in classes:
    random.shuffle(files_for_class[cls])

files_for_class = {x: files_for_class[x] for x in classes}

original_files = {'train': {'count': 14, 'files': []},
                  'val':  {'count': 3, 'files': []},
                  'test':  {'count': 3, 'files': []}}
print(f'Train video count: {original_files["train"]["count"]}')
print(f'Validate video count: {original_files["val"]["count"]}')
print(f'Test video count: {original_files["test"]["count"]}')


for split_name, split_item in original_files.items():
    split_files, files_for_class = FileOps.split_class_lists(files_for_class, split_item['count'])
    split_item['files'] = split_files

frames_count: int = FrameGenerator.get_frames_count(original_data_dir / original_files['train']['files'][0])
print(f'Frames per video: {frames_count}')

# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_fg = FrameGenerator(original_data_dir, original_files['train']['files'], frames_count, training=True)
train_ds = tf.data.Dataset.from_generator(train_fg, output_signature = output_signature)

CLASSES_COUNT = len(train_fg.class_names)

# Create the validation set
val_fg = FrameGenerator(original_data_dir, original_files['val']['files'], frames_count)
val_ds = tf.data.Dataset.from_generator(val_fg, output_signature = output_signature)

# Create the test set
test_fg = FrameGenerator(original_data_dir, original_files['test']['files'], frames_count)
test_ds = tf.data.Dataset.from_generator(test_fg, output_signature = output_signature)

# AUTOTUNE = tf.data.AUTOTUNE
# # AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
# val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
# test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)
test_ds = test_ds.batch(2)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

test_frames, test_labels = next(iter(test_ds))
print(f'Shape of test set of frames: {test_frames.shape}')
print(f'Shape of test labels: {test_labels.shape}')

with tf.device('/device:GPU:0'):
    input_shape = (None, frames_count, HEIGHT, WIDTH, 3)
    input = tf.keras.layers.Input(shape=(input_shape[1:]))
    # input = layers.Input(shape=(input_shape))
    x = input
    x = tf.keras.layers.Conv3D(filters=16,
                           kernel_size=(3, 3, 3),
                           padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # 256 -> 128
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    # 128 -> 64
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    # 64 -> 32
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    # 32 -> 16
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(CLASSES_COUNT)(x)

    model = tf.keras.Model(input, x)

    frames, label = next(iter(train_ds))
    model.build(frames)

    # Visualize the model
    tf.keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True, to_file=SOURCE_FOLDER + '/model2.png')

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                  metrics = ['accuracy'])

    history = model.fit(x = train_ds,
                        # epochs = 1,
                        epochs = 50,
                        validation_data = val_ds)
    model.evaluate(test_ds, return_dict=True)
    model.save(SOURCE_FOLDER + '/lips_recog_train.h5')

    plot_history(history)

    labels = list(train_fg.class_ids_for_name.keys())

    actual, predicted = get_actual_predicted_labels(train_ds)
    plot_confusion_matrix(actual, predicted, labels, 'training')

    actual, predicted = get_actual_predicted_labels(test_ds)
    plot_confusion_matrix(actual, predicted, labels, 'test')

    precision, recall = calculate_classification_metrics(actual, predicted, labels) # Test dataset
    print('precision: ', precision)
    print('recall: ', recall)

