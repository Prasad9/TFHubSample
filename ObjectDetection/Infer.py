import tensorflow as tf
from tqdm import tqdm
import os
import cv2
import shutil
import matplotlib.image as mpimg

from Data import Data
from Model import Model

class Infer:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']
        self._top_k = params['PLOT_TOP_K']
        self._save_path = os.path.abspath(params['INFER_PATH'] + 'Plot')

        self.data = Data(params)
        self.model = Model(params)

    def infer(self):
        shutil.rmtree(self._save_path, ignore_errors=True)
        os.mkdir(self._save_path)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            self.model.dataset.initialize_iterator(sess, self.data.infer_files)
            try:
                with tqdm(total=self.data.get_infer_data_length()) as pbar:
                    while True:
                        class_entities, boxes, file_path = sess.run([self.model.class_entities,
                                                    self.model.boxes, self.model.dataset.file_path])
                        top_class_entities = class_entities[:self._top_k]
                        top_boxes = boxes[:self._top_k]

                        self._plot_and_save(file_path[0], top_class_entities, top_boxes)
                        pbar.update(self._batch_size)
            except tf.errors.OutOfRangeError:
                pass

    def _plot_and_save(self, src_path, class_entities, boxes):
        src_path = src_path.decode('ascii')
        img = mpimg.imread(src_path)
        shape = img.shape
        for ce, b in zip(class_entities, boxes):
            ce = ce.decode('ascii')
            s1, s2, s3, s4 = int(b[1] * shape[1]), int(b[0] * shape[0]), int(b[3] * shape[1]), int(b[2] * shape[0])
            cv2.rectangle(img, (s1, s2), (s3, s4), (0, 255, 0), 10)
            cv2.putText(img, ce, (s1, s2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        dst_path = os.path.join(self._save_path, os.path.basename(src_path))
        mpimg.imsave(dst_path, img)
