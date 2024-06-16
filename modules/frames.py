import os
import random
import collections
import cv2
import numpy as np

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
