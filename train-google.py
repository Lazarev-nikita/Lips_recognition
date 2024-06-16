import os
import pathlib
import datetime
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

print(f'GPU device: {tf.test.gpu_device_name()}')

BATCH_SIZE = 4
EPOCH_COUNT = 50
#SOURCE_FOLDER = 'C:\\Projects\\my\\python\\lips_data'
SOURCE_FOLDER = '/content/drive/My Drive/lips_data'
SOURCE_VIDEO_FOLDER = SOURCE_FOLDER + '/crop-video'
CHECKPOINT_FOLDER = os.path.join(SOURCE_FOLDER, 'check-points')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, 'save_at_{epoch}')
TENSORBOARD_PATH = os.path.join(  # Timestamp included to enable timeseries graphs
    SOURCE_FOLDER, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
MODEL_SAVE_PATH = os.path.join(SOURCE_FOLDER, "lips_model")
HEIGHT = 256
WIDTH = 256
FRAMES_COUNT = 28

data_files = {'train': {'count': 12, 'files': []},
              'val':  {'count': 4, 'files': []},
              'test':  {'count': 4, 'files': []}}


def file_get_class(fname):
    return fname.split('_')[0]


def files_get_list(src_dir: str):
    files_list = os.listdir(src_dir)
    return files_list


def get_files_per_class(files_list: list):
    files_per_class_map = collections.defaultdict(list)
    for fname in files_list:
        class_name = file_get_class(fname)
        files_per_class_map[class_name].append(fname)
    return files_per_class_map


def split_class_lists(files_per_class_map: dict, count: int):
    split_files = list()
    remainder = dict()
    for cls in files_per_class_map:
        split_files.extend(files_per_class_map[cls][:count])
        remainder[cls] = files_per_class_map[cls][count:]
    return split_files, remainder


class FrameGenerator(tf.keras.utils.Sequence):
    def __init__(self, src_path: pathlib.Path, files_list: list, n_frames: int, batch_size: int, training=False):
        self.src_path = src_path
        self.files_list = files_list
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.training = training

        video_paths = list(self.src_path / f for f in self.files_list)
        classes = list(file_get_class(f) for f in self.files_list)
        self.files_pairs = list(zip(video_paths, classes))
        print(f'Pairs count: {len(self.files_pairs)}')

        self.class_names = sorted(set(classes))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
        print(f'Classes count: {len(self.class_names)}')

        self.length = int(np.floor(len(self.files_list) / self.batch_size))

    @staticmethod
    def format_frames(frame, output_size):
        resized_image = cv2.resize(frame, output_size, interpolation = cv2.INTER_CUBIC)
        resized_image = resized_image.astype(np.float32)
        resized_image /= 255.
        return resized_image

    @staticmethod
    def frames_from_video_file(video_path, n_frames, output_size=(256, 256)):
        # Read each video frame by frame
        result = [None] * n_frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return result

        for i in range(n_frames):
            ret, frame = cap.read()
            if ret:
                result[i] = FrameGenerator.format_frames(frame, output_size)
            else:
                result[i] = np.zeros_like(result[0])
        cap.release()

        result = np.array(result)[..., [2, 1, 0]]

        return result

    def on_epoch_end(self):
        if self.training:
            random.shuffle(self.files_pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images = []
        labels = []
        pairs = self.files_pairs[index * self.batch_size : (index + 1) * self.batch_size]

        for path, name in pairs:
            video_frames = FrameGenerator.frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name] # Encode labels
            images.append(video_frames)
            labels.append(label)
        return np.array(images), np.array(labels)


def create_model(classes_count: int, input_shape):
    model = tf.keras.Sequential()                               # 256*256*3*28
    model.add(tf.keras.layers.Conv3D(filters=16,                # 256*256*16*28
                                  kernel_size=(3, 3, 3),
                                  input_shape=input_shape,
                                  padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    # 256 -> 128, 28 -> 28
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

    # Block 1
    model.add(tf.keras.layers.Conv3D(filters=16,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv3D(filters=16,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    # 128 -> 64, 28 -> 14
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Block 2
    model.add(tf.keras.layers.Conv3D(filters=32,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv3D(filters=32,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    # 64 -> 32, 14 -> 7
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Block 3
    model.add(tf.keras.layers.Conv3D(filters=64,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv3D(filters=64,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    # 32 -> 16, 7 -> 7
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

    # Block 4
    model.add(tf.keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv3D(filters=128,
                                  kernel_size=(3, 3, 3),
                                  padding='same'))

    # x = tf.keras.layers.GlobalAveragePooling3D(keepdims=True)(x)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(classes_count))

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                  metrics = ['accuracy'])

    return model


def keras_model_memory_usage_in_bytes(model, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
            batch_size * shapes_mem_count
            + internal_model_mem_count
            + trainable_count
            + non_trainable_count
    )
    return total_memory

def restore_create_model(classes_count: int, input_shape):
    print('Check for checkpoints...')
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
    if latest_checkpoint is not None:
        print('Checkpoint found, restore model')
        return tf.keras.models.load_model(latest_checkpoint)
    else:
        print('Checkpoint not found, create model')
        return create_model(classes_count, input_shape)


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


callbacks = [
    # TensorBoard will store logs for each epoch and graph performance for us.
    tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=1),
    # ModelCheckpoint will save models after each epoch for retrieval later.
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH,
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto',
                                       save_weights_only = True,
                                       save_freq='epoch'),
    # EarlyStopping will terminate training when val_acc ceases to improve.
    tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                     verbose=1,
                                     patience=10,
                                     mode='auto',
                                     restore_best_weights=True),
]

original_data_dir = pathlib.Path(SOURCE_VIDEO_FOLDER)
print(f'SOURCE_VIDEO_FOLDER={SOURCE_VIDEO_FOLDER}')

video_files_list: list = files_get_list(SOURCE_VIDEO_FOLDER)
print(f'Video files found={len(video_files_list)}')

print('Extract classes...')
files_per_class = get_files_per_class(video_files_list)

classes = list(files_per_class.keys())
print(f'Video classes count: {len(classes)}')

print('Shuffle video for each class')
for cls in files_per_class:
    random.shuffle(files_per_class[cls])
files_per_class = {x: files_per_class[x] for x in classes}

print(f'Split video files to train ({data_files["train"]["count"]} per class), validate ({data_files["val"]["count"]} per class) and test ({data_files["test"]["count"]} per class) groups')
for split_name, split_item in data_files.items():
    split_files, files_per_class = split_class_lists(files_per_class, split_item['count'])
    split_item['files'] = split_files
    print(f'Files in {split_name} group: {len(split_files)}')

# create generators
print('Create Train generator...')
train_fg = FrameGenerator(original_data_dir, data_files['train']['files'], FRAMES_COUNT, BATCH_SIZE, training=True)
print('Create Validation generator...')
val_fg = FrameGenerator(original_data_dir, data_files['val']['files'], FRAMES_COUNT, BATCH_SIZE)
print('Create Test generator...')
test_fg = FrameGenerator(original_data_dir, data_files['test']['files'], FRAMES_COUNT, BATCH_SIZE)

CLASSES_COUNT = len(train_fg.class_names)
input_shape = (None, FRAMES_COUNT, HEIGHT, WIDTH, 3)

with tf.device('/device:GPU:0'):
    print('Restore or create Neuro model...')
    neuro = restore_create_model(CLASSES_COUNT, input_shape[1:])
    approximate_memory = keras_model_memory_usage_in_bytes(neuro, BATCH_SIZE)
    print(f'Approximate model memory usage = {approximate_memory}')

    # Visualize the model
    tf.keras.utils.plot_model(neuro, show_shapes=True, to_file=SOURCE_FOLDER + '/model4.png')

    print('Start training...')
    history = neuro.fit(x=train_fg,
                        validation_data=val_fg,
                        epochs=EPOCH_COUNT,
                        callbacks=callbacks)
    print('Training completed')
    plot_history(history)

    print('Evaluate on test data...')
    results = neuro.evaluate(test_fg, return_dict=True)
    print("test loss, test acc:", results)

    print("Generate predictions...")
    predictions = results.predict(test_fg)
    print("predictions shape:", predictions.shape)
