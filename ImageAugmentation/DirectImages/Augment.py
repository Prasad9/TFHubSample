import os
import shutil
import tensorflow as tf
from tqdm import tqdm
import matplotlib.image as mpimg

from Model import Model

class Augment:
    def __init__(self, params):
        self._setup_files(params['SRC_PATH'], params['DST_PATH'])
        self.augmentations_per_image = params['AUGMENTATIONS_PER_IMAGE']

        self.model = Model(params)

    def _setup_files(self, src_folder_path, dst_folder_path):
        src_folder_path = os.path.abspath(src_folder_path)
        self._dst_folder_path = os.path.abspath(dst_folder_path)
        shutil.rmtree(self._dst_folder_path, ignore_errors=True)
        os.mkdir(self._dst_folder_path)

        self._src_file_paths = os.listdir(src_folder_path)
        self._src_file_paths = [os.path.join(src_folder_path, p) for p in self._src_file_paths]

    def augment(self):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            with tqdm(total=len(self._src_file_paths) * self.augmentations_per_image) as pbar:
                for run_no in range(self.augmentations_per_image):
                    self.model.dataset.initialize_iterator(sess, self._src_file_paths)

                    try:
                        while True:
                            images, file_paths = sess.run([self.model.aug_images, self.model.dataset.file_path])
                            self._save_image(images[0], file_paths[0], run_no)
                            pbar.update(1)
                    except tf.errors.OutOfRangeError:
                        pass

    def _save_image(self, image, file_path, run_no):
        file_path = file_path.decode('ascii')
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        dst_file_path = os.path.join(self._dst_folder_path, '{}_{}.jpg'.format(file_name, run_no + 1))
        mpimg.imsave(dst_file_path, image)