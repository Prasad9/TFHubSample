import tensorflow as tf

class Dataset:
    def __init__(self, params):
        self._seq_len = params['SEQ_LEN']
        self._batch_size = params['BATCH_SIZE']

        self.data_X, self.data_y = self._prepare_dataset()
        self.data_X = tf.identity(self.data_X, name='data_X')
        self.data_y = tf.identity(self.data_y, name='data_y')

    def _prepare_dataset(self):
        self._pl_X = tf.placeholder(tf.string, [None], name='pl_X')
        self._pl_y = tf.placeholder(tf.int32, [None], name='pl_y')
        dataset = tf.data.Dataset.from_tensor_slices((self._pl_X, self._pl_y))
        dataset = dataset.map(self._fetch_videos)
        dataset = dataset.batch(self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        self._iterator_initializer = iterator.make_initializer(dataset, name='initializer')
        data_X, data_y = iterator.get_next()
        return data_X, data_y

    def _fetch_videos(self, batch_X, batch_y):
        video_data = []
        for frame_index in range(self._seq_len):
            file_path = tf.strings.join([batch_X, tf.constant('/{:05d}.jpg'.format(frame_index))])
            X_data = tf.read_file(file_path)
            X_data = tf.image.decode_jpeg(X_data)
            X_data = tf.expand_dims(X_data, axis=0)
            video_data.append(X_data)
        X_data = tf.concat(video_data, axis=0)
        X_data = tf.cast(X_data, tf.float32)
        X_data = tf.divide(X_data, tf.constant(255.0, dtype=tf.float32))
        return X_data, batch_y

    def initialize_iterator(self, sess, X_data, y_data):
        feed_dict = {
            self._pl_X: X_data,
            self._pl_y: y_data
        }
        sess.run(self._iterator_initializer, feed_dict=feed_dict)

    def initialize_test_iterator_for_saved_model_graph(self, sess, X_data, y_data):
        pl_X = sess.graph.get_tensor_by_name('pl_X:0')
        pl_y = sess.graph.get_tensor_by_name('pl_y:0')
        initializer = sess.graph.get_operation_by_name('initializer')

        feed_dict = {
            pl_X: X_data,
            pl_y: y_data
        }
        sess.run(initializer, feed_dict=feed_dict)
