import tensorflow as tf
import shutil
import os
from tqdm import tqdm
import csv

from Model import Model
from Data import Data

class Train:
    def __init__(self, params):
        self._epochs = params['EPOCHS']
        self._batch_size = params['BATCH_SIZE']
        self._lr = params['LEARNING_RATE']
        self._n_class = params['N_CLASS']

        self.data = Data(params)
        self.model = Model(params)

        self._save_path = os.path.abspath('./Model')

    def train(self):
        shutil.rmtree(self._save_path, ignore_errors=True)
        os.mkdir(self._save_path)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            current_lr = self._lr

            for epoch_no in range(self._epochs):
                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0

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

    def test(self):
        test_graph = tf.Graph()
        with test_graph.as_default():
            with tf.Session(graph=test_graph) as sess, \
                    open(os.path.join(self._save_path, 'results.csv'), 'w') as fid:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                saved_model_path = os.path.join(self._save_path, str(self._epochs - 1))
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
                self.model.dataset.initialize_test_iterator_for_saved_model_graph(sess, self.data.X_test_id,
                                                                    self.data.X_test)

                csv_fid = csv.writer(fid)
                csv_fid.writerow(['PhraseId', 'Sentiment'])

                predictions_op = test_graph.get_tensor_by_name('predictions:0')
                text_id_op = test_graph.get_tensor_by_name('text_id:0')
                try:
                    with tqdm(total=self.data.get_test_data_length()) as pbar:
                        while True:
                            predictions, phrase_id = sess.run([predictions_op, text_id_op])
                            predictions = predictions.tolist()
                            phrase_id = phrase_id.tolist()

                            for pred, p in zip(predictions, phrase_id):
                                csv_fid.writerow([p, pred])
                            pbar.update(self._batch_size)
                except tf.errors.OutOfRangeError:
                    pass

    def _save_model(self, sess, epoch_no):
        inputs = {
            'pl_phrase_id': sess.graph.get_tensor_by_name('pl_phrase_id:0'),
            'pl_phrase_text': sess.graph.get_tensor_by_name('pl_phrase_text:0'),
            'pl_sentiment': sess.graph.get_tensor_by_name('pl_sentiment:0')
        }
        outputs = {'accuracy': self.model.accuracy}

        export_dir = os.path.join(self._save_path, str(epoch_no))
        tf.saved_model.simple_save(sess, export_dir, inputs, outputs)