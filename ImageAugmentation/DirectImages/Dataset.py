import tensorflow as tf

class Dataset:
    def __init__(self):
        self._create_iterator()

    def _create_iterator(self):
        self._image_file_path = tf.placeholder(tf.string, (None))
        dataset = tf.data.Dataset.from_tensor_slices(self._image_file_path)
        dataset = dataset.map(self._map_files)

        # Batching is being done 1 image per batch
        # Images may be of different sizes and hence cannot be batched collectively
        dataset = dataset.batch(1)

        self._iterator = dataset.make_initializable_iterator()
        self.img_data, self.file_path = self._iterator.get_next()

    def _map_files(self, file_path):
        img_data = tf.read_file(file_path)
        img_data = tf.image.decode_jpeg(img_data)
        img_data = tf.cast(img_data, tf.float32)
        img_data = tf.divide(img_data, tf.constant(255.0, dtype=tf.float32))
        return img_data, file_path

    def initialize_iterator(self, sess, file_paths):
        feed_dict = {
            self._image_file_path: file_paths
        }
        sess.run(self._iterator.initializer, feed_dict=feed_dict)
