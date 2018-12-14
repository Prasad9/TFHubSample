import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, params):
        self._n_class = params['N_CLASS']
        self._batch_size = params['BATCH_SIZE']

        self._create_iterator()

    def _create_iterator(self):
        self._pl_image_file_paths = tf.placeholder(tf.string, (None), name='pl_image_paths')
        self._pl_labels = tf.placeholder(tf.int32, (None, self._n_class), name='pl_labels')

        dataset = tf.data.Dataset.from_tensor_slices((self._pl_image_file_paths, self._pl_labels))
        dataset = dataset.map(self._map_files)
        dataset = dataset.batch(self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        self._iterator_initializer = iterator.make_initializer(dataset, name='initializer')
        self.img_data, self.file_path, self.labels = iterator.get_next()
        self.file_path = tf.identity(self.file_path, name='file_path')

    def _map_files(self, file_path, multilabels):
        img_data = tf.read_file(file_path)
        return img_data, file_path, multilabels

    def initialize_iterator(self, sess, file_paths, multilabels):
        feed_dict = {
            self._pl_image_file_paths: file_paths,
            self._pl_labels: multilabels
        }
        sess.run(self._iterator_initializer, feed_dict=feed_dict)

    def initialize_test_iterator_for_saved_model_graph(self, sess, X_data):
        pl_X = sess.graph.get_tensor_by_name('pl_image_paths:0')
        pl_y = sess.graph.get_tensor_by_name('pl_labels:0')
        initializer = sess.graph.get_operation_by_name('initializer')

        y_data = np.zeros((len(X_data), self._n_class), dtype=np.float32)
        feed_dict = {
            pl_X: X_data,
            pl_y:  y_data
        }
        sess.run(initializer, feed_dict=feed_dict)
