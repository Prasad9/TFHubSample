import os
import shutil
from tqdm import tqdm
import cv2

class PreprocessData:
    def __init__(self, paths):
        self.paths = paths

    def resize_data(self, size):
        for path in self.paths:
            print('Resizing {}'.format(path))
            src_path = os.path.abspath(path)
            dst_path = src_path + 'Resize'
            shutil.rmtree(dst_path, ignore_errors=True)
            os.mkdir(dst_path)

            filenames = os.listdir(src_path)
            for filename in tqdm(filenames):
                src_filepath = os.path.join(src_path, filename)
                dst_filepath = os.path.join(dst_path, filename)

                try:
                    img = cv2.imread(src_filepath)
                    img = cv2.resize(img, size)
                    cv2.imwrite(dst_filepath, img)
                except:
                    pass

if __name__ == '__main__':
    resize_paths = ['./Data/train', './Data/test']    # Give the path where the data folders are present
    p = PreprocessData(resize_paths)
    p.resize_data((224, 224))          # Specify the size you would like to resize the images to