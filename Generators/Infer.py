import tensorflow as tf
from tqdm import tqdm
import os
import shutil
import matplotlib.image as mpimg

from Data import Data
from Model import Model

class Infer:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']
        self._save_path = os.path.abspath(params['SAVE_FOLDER'])
        self._save_index_at = 0

        self.data = Data(params)
        self.model = Model(params)

    def infer(self):
        shutil.rmtree(self._save_path, ignore_errors=True)
        os.mkdir(self._save_path)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            self.model.dataset.initialize_iterator(sess, self.data.total_images)
            try:
                with tqdm(total=self.data.total_images) as pbar:
                    while True:
                        images = sess.run(self.model.images)
                        self._save_images(images)
                        pbar.update(self._batch_size)
            except tf.errors.OutOfRangeError:
                pass

    def _save_images(self, images):
        for save_index, image in enumerate(images, self._save_index_at):
            save_path = os.path.join(self._save_path, '{:05d}.jpg'.format(save_index))
            mpimg.imsave(save_path, image)

        self._save_index_at += len(images)