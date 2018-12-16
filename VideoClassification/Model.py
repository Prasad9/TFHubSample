import tensorflow as tf
import tensorflow_hub as hub
from Dataset import Dataset

class Model:
    def __init__(self, params, n_class):
        self._n_class = n_class
        self._mean = 0.0
        self._stddev = 0.1
        self._depth = params['SEQ_LEN']
        self._width = params['IMG_WIDTH'],
        self._height = params['IMG_HEIGHT']
        self._build_architecture(params)

    def _build_model(self, images):
        self.module = hub.Module("https://tfhub.dev/deepmind/i3d-kinetics-400/1", trainable=True)
        #print(module.get_signature_names())
        features = self.module(dict(rgb_input=images))
        #print(features)

        with tf.variable_scope('CustomLayer'):
            mean = 0.0
            stddev = 0.1
            weight = tf.get_variable('weights',
                                     initializer=tf.truncated_normal((400, self._n_class), mean=mean,
                                                                     stddev=stddev, seed=189))
            bias = tf.get_variable('bias', initializer=tf.ones((self._n_class)))
            logits = tf.nn.xw_plus_b(features, weight, bias)
            #print(logits)

        return logits

    def _build_architecture(self, params):
        tf.reset_default_graph()

        self.dataset = Dataset(params)
        self.lr = tf.placeholder(tf.float32, ())
        one_hot_y = tf.one_hot(self.dataset.data_y, self._n_class, dtype=tf.int32)
        self.logits = self._build_model(self.dataset.data_X)
        self.logits = tf.identity(self.logits, name='logits')
        self.predictions = tf.argmax(self.logits, axis=1, output_type=tf.int32, name='predictions')

        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=self.logits)
        self.loss = tf.reduce_sum(softmax)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.dataset.data_y), tf.float32),
                                       name='accuracy')
