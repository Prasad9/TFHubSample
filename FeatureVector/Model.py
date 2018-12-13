import tensorflow as tf
import tensorflow_hub as hub

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._n_class = params['N_CLASS']
        self._prepare_graph(params)

    def _prepare_graph(self, params):
        tf.reset_default_graph()
        self.lr = tf.placeholder(tf.float32, shape=())

        self.dataset = Dataset(params)

        logits = self._prepare_model(self.dataset.img_data)
        float_y = tf.cast(self.dataset.labels, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=float_y)
        self.loss = tf.reduce_sum(cross_entropy)

        sigmoid_logits = tf.nn.sigmoid(logits)
        self.predictions = tf.cast(tf.round(sigmoid_logits), tf.int32, name='predictions')
        self.accuracy = tf.reduce_sum(tf.reduce_min(tf.cast(tf.equal(self.predictions, self.dataset.labels),
                                                                tf.float32), axis=1))

        self._prepare_optimizer_stage(fine_tune_upto=1)

    def _prepare_model(self, images):
        module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
                            trainable=True) # Trainable is True since we are going to fine-tune the model
        module_features = module(dict(images=images), signature="image_feature_vector",
                                 as_dict=True)
        features = module_features["default"]
        print(features)

        with tf.variable_scope('CustomLayer'):
            mean = 0.0
            stddev = 0.1
            weight = tf.get_variable('weights',
                                     initializer=tf.truncated_normal((2048, self._n_class), mean=mean,
                                                                     stddev=stddev, seed=198))
            bias = tf.get_variable('bias', initializer=tf.ones((self._n_class)))
            logits = tf.nn.xw_plus_b(features, weight, bias)
            print(logits)

        return logits

    def _prepare_optimizer_stage(self, fine_tune_upto):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CustomLayer')
        if fine_tune_upto < 2:
            var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module/resnet_v2_50/block4')
            var_list.extend(var_list2)
        if fine_tune_upto == 2:
            var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module/resnet_v2_50/block3')
            var_list.extend(var_list2)

        print('Var list to Optimise:')
        print(*var_list, sep='\n')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=var_list)
