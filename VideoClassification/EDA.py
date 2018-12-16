import os
from tqdm import tqdm

class EDA:
    def __init__(self):
        pass

    @staticmethod
    def get_avg_frames(folder):
        total_frames = 0
        clips_path = os.path.abspath(folder)
        clips = os.listdir(clips_path)
        for clip in tqdm(clips):
            clip_path = os.path.join(clips_path, clip)
            total_frames += len(os.listdir(clip_path))
        print('Average number of frames is {:.2f}'.format(total_frames / len(clips)))


if __name__ == '__main__':
    EDA.get_avg_frames('./Data/20bn-jester-v1')
