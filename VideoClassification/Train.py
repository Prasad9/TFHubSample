import tensorflow as tf
from tqdm import tqdm
import os
import shutil
import csv

from Data import Data
from Model import Model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class Train:
    def __init__(self, params):
        self._epochs = params['EPOCHS']
        self._batch_size = params['BATCH_SIZE']
        self._lr = params['LEARNING_RATE']
        self._divide_lr = params['DIVIDE_LEARNING_RATE_AT']

        self.data = Data(params)
        n_class = len(params['REQD_LABELS'])
        self.model = Model(params, n_class=n_class)
        self._save_path = os.path.abspath('./Model')

    def train(self):
        shutil.rmtree(self._save_path, ignore_errors=True)
        os.mkdir(self._save_path)
        current_lr = self._lr

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            for epoch_no in range(self._epochs):
                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0
                if epoch_no in self._divide_lr:
                    current_lr /= 10

                print('\nEpoch: {}, lr: {:.6f}'.format(epoch_no + 1, current_lr))
                self.model.dataset.initialize_iterator(sess, self.data.X_train, self.data.y_train)
                try:
                    with tqdm(total=self.data.get_train_data_length()) as pbar:
                        while True:
                            _, l, a = sess.run([self.model.optimizer, self.model.loss, self.model.accuracy],
                                               feed_dict={self.model.lr: current_lr})
                            train_loss += l
                            train_accuracy += a
                            pbar.update(self._batch_size)
                except tf.errors.OutOfRangeError:
                    pass

                self.model.dataset.initialize_iterator(sess, self.data.X_val, self.data.y_val)
                try:
                    with tqdm(total=self.data.get_val_data_length()) as pbar:
                        while True:
                            l, a = sess.run([self.model.loss, self.model.accuracy])
                            val_loss += l
                            val_accuracy += a
                            pbar.update(self._batch_size)
                except tf.errors.OutOfRangeError:
                    pass

                train_accuracy /= self.data.get_train_data_length()
                train_loss /= self.data.get_train_data_length()
                val_accuracy /= self.data.get_val_data_length()
                val_loss /= self.data.get_val_data_length()

                print('Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy, train_loss))
                print('Validation accuracy: {:.4f}, loss: {:.4f}'.format(val_accuracy, val_loss))

                self._save_model(sess, epoch_no)

    def test(self, ):
        test_graph = tf.Graph()
        with test_graph.as_default():
            with tf.Session(graph=test_graph) as sess,\
                    open(os.path.join(self._save_path, 'results.csv'), 'w') as fid:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                saved_model_path = os.path.join(self._save_path, str(self._epochs - 1))
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
                self.model.dataset.initialize_test_iterator_for_saved_model_graph(sess,
                                                            self.data.X_test, self.data.X_test_filenames)

                csv_fid = csv.writer(fid, delimiter=';')
                logits_op = test_graph.get_tensor_by_name('logits:0')
                predictions_op = tf.argmax(logits_op, axis=1)
                data_y_op = test_graph.get_tensor_by_name('data_y:0')
                try:
                    with tqdm(total=self.data.get_test_data_length()) as pbar:
                        while True:
                            predictions, filenames = sess.run([predictions_op, data_y_op])

                            for pred, filename in zip(predictions, filenames):
                                label = self.data.label_at_index(pred)
                                csv_fid.writerow([filename, label])
                            pbar.update(self._batch_size)
                except tf.errors.OutOfRangeError:
                    pass

    def _save_model(self, sess, epoch_no):
        inputs = {'pl_X': sess.graph.get_tensor_by_name('pl_X:0'),
                  'pl_y': sess.graph.get_tensor_by_name('pl_y:0')}
        outputs = {'accuracy': self.model.accuracy}

        export_dir = os.path.join(self._save_path, str(epoch_no))
        tf.saved_model.simple_save(sess, export_dir, inputs, outputs)
