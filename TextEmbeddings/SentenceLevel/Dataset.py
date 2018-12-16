import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, params):
        self._n_class = params['N_CLASS']
        self._batch_size = params['BATCH_SIZE']

        self._create_iterator()

    def _create_iterator(self):
        self._pl_phrase_id = tf.placeholder(tf.int32, (None), name='pl_phrase_id')
        self._pl_phrase_text = tf.placeholder(tf.string, (None), name='pl_phrase_text')
        self._pl_sentiment = tf.placeholder(tf.int32, (None), name='pl_sentiment')

        dataset = tf.data.Dataset.from_tensor_slices((self._pl_phrase_id, self._pl_phrase_text,
                                                      self._pl_sentiment))
        dataset = dataset.batch(self._batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        self._iterator_initializer = iterator.make_initializer(dataset, name='initializer')
        self.text_id, self.text_data, self.labels = iterator.get_next()
        self.text_id = tf.identity(self.text_id, name='text_id')

    def initialize_iterator(self, sess, phrase_text, sentiment):
        phrase_id = np.zeros((len(phrase_text)), dtype=np.int32)
        feed_dict = {
            self._pl_phrase_id: phrase_id,
            self._pl_phrase_text: phrase_text,
            self._pl_sentiment: sentiment
        }
        sess.run(self._iterator_initializer, feed_dict=feed_dict)

    def initialize_test_iterator_for_saved_model_graph(self, sess, phrase_id, phrase_text):
        pl_phrase_id = sess.graph.get_tensor_by_name('pl_phrase_id:0')
        pl_phrase_text = sess.graph.get_tensor_by_name('pl_phrase_text:0')
        pl_sentiment = sess.graph.get_tensor_by_name('pl_sentiment:0')
        initializer = sess.graph.get_operation_by_name('initializer')

        sentiment = np.zeros((len(phrase_text)), dtype=np.float32)
        feed_dict = {
            pl_phrase_id: phrase_id,
            pl_phrase_text: phrase_text,
            pl_sentiment: sentiment
        }
        sess.run(initializer, feed_dict=feed_dict)
