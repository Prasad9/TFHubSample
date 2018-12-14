import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.contrib.image

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._n_class = params['N_CLASS']
        self._img_size = params['IMAGE_SIZE']
        self._prepare_graph(params)

    def _prepare_graph(self, params):
        tf.reset_default_graph()
        self.lr = tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.dataset = Dataset(params)

        logits = self._prepare_model(self.dataset.img_data, self.is_training)
        float_y = tf.cast(self.dataset.labels, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=float_y)
        self.loss = tf.reduce_sum(cross_entropy)

        sigmoid_logits = tf.nn.sigmoid(logits)
        self.predictions = tf.cast(tf.round(sigmoid_logits), tf.int32, name='predictions')
        self.accuracy = tf.reduce_sum(tf.reduce_min(tf.cast(tf.equal(self.predictions, self.dataset.labels),
                                                            tf.float32), axis=1))

        self._prepare_optimizer_stage(fine_tune_upto=1)

    def _prepare_model(self, images, is_training):
        aug_module = hub.Module('https://tfhub.dev/google/image_augmentation/crop_color/1')
        # print(aug_module)
        # print(aug_module.get_signature_names())
        aug_features = aug_module(dict(encoded_images=images,
                                       image_size=self._img_size,
                                       augmentation=is_training))
        # print(aug_features)

        resnet_module = hub.Module('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
                                   trainable=True)  # Trainable is True since we are going to fine-tune the model

        img_features = resnet_module(dict(images=aug_features))

        with tf.variable_scope('CustomLayer'):
            mean = 0.0
            stddev = 0.1
            weight = tf.get_variable('weights',
                                     initializer=tf.truncated_normal((2048, self._n_class), mean=mean,
                                                                     stddev=stddev, seed=198))
            bias = tf.get_variable('bias', initializer=tf.ones((self._n_class)))
            logits = tf.nn.xw_plus_b(img_features, weight, bias)
            print(logits)

        return logits

    def _prepare_optimizer_stage(self, fine_tune_upto):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CustomLayer')
        if fine_tune_upto < 2:
            var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module_1/resnet_v2_50/block4')
            var_list.extend(var_list2)
        if fine_tune_upto == 2:
            var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module_1/resnet_v2_50/block3')
            var_list.extend(var_list2)

        print('Var list to Optimise:')
        print(*var_list, sep='\n')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list=var_list)
