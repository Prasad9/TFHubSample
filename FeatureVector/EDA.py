import os
import cv2
from tqdm import tqdm

class EDA:
    def __init__(self):
        pass

    @staticmethod
    def average_size(path):
        width, height = 0, 0
        path = os.path.abspath(path)
        files = os.listdir(path)
        for f in tqdm(files):
            img = cv2.imread(os.path.join(path, f))
            width += img.shape[0]
            height += img.shape[1]

        print('Width = {:.2f}, height = {:.2f}'.format(width / len(files), height / len(files)))


if __name__ == '__main__':
    EDA.average_size('./Data/train')
    EDA.average_size('./Data/test')