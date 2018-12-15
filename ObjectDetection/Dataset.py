import tensorflow as tf

class Dataset:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']

        self._create_iterator()

    def _create_iterator(self):
        self._pl_image_file_paths = tf.placeholder(tf.string, (None), name='pl_image_paths')

        dataset = tf.data.Dataset.from_tensor_slices(self._pl_image_file_paths)
        dataset = dataset.map(self._map_files)
        dataset = dataset.batch(self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        self._iterator_initializer = iterator.make_initializer(dataset, name='initializer')
        self.img_data, self.file_path = iterator.get_next()

    def _map_files(self, file_path):
        img_data = tf.read_file(file_path)
        img_data = tf.image.decode_jpeg(img_data)
        img_data = tf.cast(img_data, tf.float32)
        img_data = tf.divide(img_data, tf.constant(255.0, dtype=tf.float32))
        return img_data, file_path

    def initialize_iterator(self, sess, file_paths):
        feed_dict = {
            self._pl_image_file_paths: file_paths
        }
        sess.run(self._iterator_initializer, feed_dict=feed_dict)
