import os
import pathlib
import datetime
import random
import collections
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

print(f'GPU device: {tf.test.gpu_device_name()}')

TPU_ENABLED = False
GPU_ENABLED = True

BATCH_SIZE = 2
EPOCH_COUNT = 50
#SOURCE_FOLDER = 'C:\\Projects\\my\\python\\lips_data'
SOURCE_FOLDER = '/content/drive/My Drive/lips_data'
# SOURCE_VIDEO_FOLDER = SOURCE_FOLDER + '/crop-video'
SOURCE_VIDEO_FOLDER = SOURCE_FOLDER + '/crop-video-2'
STORE_CHECKPOINT_FOLDER = os.path.join(SOURCE_FOLDER, 'check-points')
RESTORE_CHECKPOINT_FOLDER = os.path.join(SOURCE_FOLDER, 'check-points-backup')
CHECKPOINT_FOLDER_MASK = 'save_at_{epoch}'
CHECKPOINT_PATH = os.path.join(STORE_CHECKPOINT_FOLDER, CHECKPOINT_FOLDER_MASK)
SETTINGS_PATH = os.path.join(SOURCE_FOLDER, 'train-data.json')
TENSORBOARD_PATH = os.path.join(  # Timestamp included to enable timeseries graphs
    SOURCE_FOLDER, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
MODEL_SAVE_PATH = os.path.join(SOURCE_FOLDER, "lips_model")
HEIGHT = 128
WIDTH = 128
FRAMES_COUNT = 5
CHANNELS = 3

data_files = {'train': {'count': 30, 'files': []},
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


class Settings:
    path: str = None
    config: dict = None

    def __init__(self, path):
        self.config = {}
        self.path = path
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.config = json.load(f)

    def get(self):
        return self.config

    def set(self, config):
        if (config is None):
            config = {}

        self.config = config
        self.save()

    def has_value(self, name):
        return name in self.config

    def get_value(self, name: str, def_value: any = None):
        return self.get_value_cfg(self.config, name, def_value)

    def get_value_cfg(self, cfg: dict, name: str, def_value: any = None):
        if name not in cfg:
            cfg[name] = def_value
        return cfg[name]

    def set_value(self, name: str, value: any):
        self.config[name] = value
        self.save()

    def save(self):
        with open(self.path, 'w', encoding="utf-8") as fo:
            fo.write(json.dumps(self.config, indent=2, ensure_ascii=False))


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
        h, w, c = frame.shape
        frm = None
        if CHANNELS == 1:
            grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            if output_size[0] != w or output_size[1] != h:
                resized = cv2.resize(grayFrame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
                frm = resized
            else:
                frm = grayFrame
        else:
            if output_size[0] != w or output_size[1] != h:
                resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
                frm = resized
            else:
                frm = frame

        resized_image = frm.astype(np.float32)
        resized_image /= 255.
        return resized_image

    @staticmethod
    def frames_from_video_file(video_path, n_frames, output_size=(WIDTH, HEIGHT)):
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

        if CHANNELS == 1:
            result = np.array(result)
        else:
            result = np.array(result)[..., [2, 1, 0]]
        return result

    def on_epoch_end(self):
        if self.training:
            random.shuffle(self.files_pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # print(f'__getitem__({index})')
        images = []
        labels = []
        pairs = self.files_pairs[index * self.batch_size : (index + 1) * self.batch_size]

        for path, name in pairs:
            video_frames = FrameGenerator.frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name] # Encode labels
            images.append(video_frames)
            labels.append(label)

        np_img = np.array(images)
        np_labels = np.array(labels)
        # print(f'images shape={np_img.shape}; labels shape={np_labels}')
        return np_img, np_labels


def create_model(classes_count: int, input_shape):
    model = tf.keras.Sequential()                               # 128*128*3*5

    # Block 1
    model.add(tf.keras.layers.Conv3D(filters=64,                # 128*128*64*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=64,                # 128*128*64*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))

    # Block 2
    # 128 -> 64, 5 -> 5
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(filters=128,                # 64*64*128*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=128,                # 64*64*128*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))

    # Block 3
    # 64 -> 32, 5 -> 5
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(filters=256,                # 32*32*256*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=256,                # 32*32*256*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))

    # Block 4
    # 32 -> 16, 5 -> 5
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(tf.keras.layers.Conv3D(filters=512,                # 16*16*512*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=512,                # 16*16*512*5
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))

    # Block 5
    # 16 -> 8, 5 -> 5
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(tf.keras.layers.Conv3D(filters=512,                # 8*8*512*2
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=512,                # 8*8*512*2
                                     kernel_size=(3, 3, 3),
                                     input_shape=input_shape,
                                     padding='same',
                                     activation='relu'))

    model.add(tf.keras.layers.Flatten())                        # 1*1*65536
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(4096,activation='relu'))    # 1*1*4096
    model.add(tf.keras.layers.Dense(4096,activation='relu'))    # 1*1*4096
    model.add(tf.keras.layers.Dense(classes_count))             # 1*1*<classes-count>

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                  metrics = ['accuracy'])

    return model


def create_model_2(classes_count: int, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1),activation='relu', padding='same'), input_shape=input_shape))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1),activation='relu', padding='same')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2, 2)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1),activation='relu', padding='same')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1),activation='relu', padding='same')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2,2)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(32,return_sequences=False,dropout=0.2)) # used 32 units

    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(classes_count))
    # model.summary()

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
    # return create_model_2(classes_count, input_shape)
    #
    latest_checkpoint = None
    print('Check for checkpoints...')
    maxIdx = -1
    if os.path.exists(RESTORE_CHECKPOINT_FOLDER):
        chpts = os.listdir(RESTORE_CHECKPOINT_FOLDER)
        for d_epoch in chpts:
            elems = d_epoch.split('_')
            maxIdx = max(maxIdx, int(elems[2]))
    if maxIdx > -1:
        ch_folder = CHECKPOINT_FOLDER_MASK.format(epoch = maxIdx)
        path = os.path.join(RESTORE_CHECKPOINT_FOLDER, ch_folder)

        print(f'Found folder: {path}')
        return tf.keras.models.load_model(path)
        #latest_checkpoint = tf.train.latest_checkpoint(path)
        #print('latest_checkpoint=', latest_checkpoint)

    if latest_checkpoint is not None:
        print('Checkpoint found, restore model')
        return tf.keras.models.load_model(latest_checkpoint)
    else:
        print('Checkpoint not found, create model')
        # return create_model(classes_count, input_shape)
        return create_model_2(classes_count, input_shape)


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
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto',
                                       save_weights_only = False,
                                       save_freq='epoch'),
    # EarlyStopping will terminate training when val_acc ceases to improve.
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
]

#################################################################################
#################################################################################
trainSettings = Settings(SETTINGS_PATH)

original_data_dir = pathlib.Path(SOURCE_VIDEO_FOLDER)
print(f'SOURCE_VIDEO_FOLDER={SOURCE_VIDEO_FOLDER}')

if trainSettings.has_value('data_files'):
    print(f'Load training files from config...')
    data_files = trainSettings.get_value('data_files')
else:
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

    trainSettings.set_value('data_files', data_files)


# create generators
print('Create Train generator...')
train_fg = FrameGenerator(original_data_dir, data_files['train']['files'], FRAMES_COUNT, BATCH_SIZE, training=True)
print('Create Validation generator...')
val_fg = FrameGenerator(original_data_dir, data_files['val']['files'], FRAMES_COUNT, BATCH_SIZE)
print('Create Test generator...')
test_fg = FrameGenerator(original_data_dir, data_files['test']['files'], FRAMES_COUNT, BATCH_SIZE)

CLASSES_COUNT = len(train_fg.class_names)
input_shape = (None, FRAMES_COUNT, HEIGHT, WIDTH, CHANNELS)

neuro = None

if TPU_ENABLED:
    print('TPU shall be enabled, try to connect...')
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.TPUStrategy(tpu)

    # creating the model in the TPUStrategy scope means we will train the model on the TPU
    with tpu_strategy.scope():
        neuro = restore_create_model(CLASSES_COUNT, input_shape[1:])
elif GPU_ENABLED:
    print('GPU shall be enabled, try to connect...')
    with tf.device('/device:GPU:0'):
        print('Restore or create Neuro model...')
        neuro = restore_create_model(CLASSES_COUNT, input_shape[1:])
else:
    print('CPU available only')
    neuro = restore_create_model(CLASSES_COUNT, input_shape[1:])

approximate_memory = keras_model_memory_usage_in_bytes(neuro, BATCH_SIZE)
print(f'Approximate model memory usage = {approximate_memory}')

# Visualize the model
tf.keras.utils.plot_model(neuro, show_shapes=True, to_file=SOURCE_FOLDER + '/model5.png')

print('Start training...')
history = neuro.fit(x=train_fg,
                    validation_data=val_fg,
                    epochs=EPOCH_COUNT,
                    callbacks=callbacks)
print('Training completed')
print('Save model...')
neuro.save(MODEL_SAVE_PATH)
plot_history(history)

print('Evaluate on test data...')
results = neuro.evaluate(test_fg, return_dict=True)
print("test loss, test acc:", results)

print("Generate predictions...")
predictions = neuro.predict(test_fg)
print("predictions shape:", predictions.shape)
