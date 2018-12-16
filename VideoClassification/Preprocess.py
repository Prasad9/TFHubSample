import cv2
import shutil
import os
from tqdm import tqdm

class Preprocess:
    def __init__(self, params):
        self.src_path = os.path.abspath(params['SRC_PATH'])
        self.dst_path = os.path.abspath(params['DST_PATH'])

    def resize(self, size, seq_length):
        shutil.rmtree(self.dst_path, ignore_errors=True)
        os.mkdir(self.dst_path)

        folders = os.listdir(self.src_path)
        for folder in tqdm(folders):
            src_folder_path = os.path.join(self.src_path, folder)
            dst_folder_path = os.path.join(self.dst_path, folder)
            os.mkdir(dst_folder_path)

            files = os.listdir(src_folder_path)
            files.sort()
            diff = seq_length - len(files)
            counter = 0
            if diff < 0:
                files = files[:seq_length]
            elif diff > 0:
                src_file_path = os.path.join(src_folder_path, files[0])
                img = cv2.imread(src_file_path)
                img = cv2.resize(img, size)
                for diff_at in range(diff):
                    dst_file_path = os.path.join(dst_folder_path, '{:05d}.jpg'.format(diff_at))
                    cv2.imwrite(dst_file_path, img)
                counter = diff

            for counter_at, file in enumerate(files, counter):
                src_file_path = os.path.join(src_folder_path, file)
                dst_file_path = os.path.join(dst_folder_path, '{:05d}.jpg'.format(counter_at))
                img = cv2.imread(src_file_path)
                img = cv2.resize(img, size)
                cv2.imwrite(dst_file_path, img)


if __name__ == '__main__':
    params = {
        'SRC_PATH': './Data/20bn-jester-v1',        # Path of downloaded videos
        'DST_PATH': './Data/VideosResized224'       # Path where you wish to generate the videos
    }
    p = Preprocess(params)
    p.resize((224, 224), 36)         # Resized video and number of frames in each video clip
