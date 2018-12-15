import tensorflow as tf
from tqdm import tqdm
import os

from Data import Data
from Model import Model

class Infer:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']

        self.data = Data(params)
        self.model = Model(params)

    def infer(self):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            self.model.dataset.initialize_iterator(sess, self.data.infer_files)
            try:
                with tqdm(total=self.data.get_infer_data_length()) as pbar:
                    while True:
                        top_pred, file_path = sess.run([self.model.top_prediction, self.model.dataset.file_path])
                        probabilities, class_ids = top_pred.values, top_pred.indices

                        for p, c, fp in zip(probabilities, class_ids, file_path):
                            fp = os.path.basename(fp.decode('ascii'))
                            print('File: ', fp)
                            for p_i, c_i in zip(p, c):
                                print('Class: {}, probability: {:.4f}'.format(self.data.classes[c_i], p_i))
                            print()
                        pbar.update(self._batch_size)
            except tf.errors.OutOfRangeError:
                pass


