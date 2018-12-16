import tensorflow as tf

class Dataset:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']

        self._create_iterator()

    def _create_iterator(self):
        self._pl_image_count = tf.placeholder(tf.int32, (None))

        dataset = tf.data.Dataset.from_tensor_slices(self._pl_image_count)
        dataset = dataset.map(self._map_latent_space)
        dataset = dataset.batch(self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        self._iterator_initializer = iterator.make_initializer(dataset, name='initializer')
        self.latent_vector_space = iterator.get_next()

    def _map_latent_space(self, total_images):
        mean = 0.0
        stddev = 1.0
        latent_vector_space = tf.random_normal((total_images, 512), mean=mean, stddev=stddev)
        return latent_vector_space

    def initialize_iterator(self, sess, image_count):
        feed_dict = {
            self._pl_image_count: [image_count]
        }
        sess.run(self._iterator_initializer, feed_dict=feed_dict)