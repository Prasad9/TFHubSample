import tensorflow as tf
import tensorflow_hub as hub

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._n_class = params['N_CLASS']
        self._max_sentence_words = params['MAX_SENTENCE_WORDS']
        self._prepare_graph(params)

    def _prepare_graph(self, params):
        tf.reset_default_graph()

        self.dataset = Dataset(params)

        self.lr = tf.placeholder(tf.float32, shape=())
        one_hot_y = tf.one_hot(self.dataset.labels, depth=self._n_class)
        logits = self._prepare_model(self.dataset.text_data, self.dataset.text_len)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
        self.loss = tf.reduce_sum(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self.predictions = tf.argmax(logits, axis=1, output_type=tf.int32, name='predictions')
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.dataset.labels), tf.float32))

    def _prepare_model(self, tokens, sequence_len):
        module = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)
        module_features = module(dict(tokens=tokens, sequence_len=sequence_len),
                                 signature='tokens', as_dict=True)
        embeddings = module_features["elmo"]
        print(embeddings)

        with tf.variable_scope('Layer1'):
            cell_fw1 = tf.nn.rnn_cell.LSTMCell(num_units=128, state_is_tuple=True)
            cell_bw1 = tf.nn.rnn_cell.LSTMCell(num_units=128, state_is_tuple=True)

            outputs1, states1 = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw1,
                cell_bw=cell_bw1,
                inputs=embeddings,
                dtype=tf.float32)

        rnn_output = tf.reshape(outputs1[0], (-1, 128 * self._max_sentence_words))

        with tf.variable_scope('Layer2'):
            weight2 = tf.get_variable('weight', initializer=tf.truncated_normal((128 * self._max_sentence_words,
                                                                                 self._n_class)))
            bias2 = tf.get_variable('bias', initializer=tf.ones(self._n_class))
            logits = tf.nn.xw_plus_b(rnn_output, weight2, bias2)
        return logits
