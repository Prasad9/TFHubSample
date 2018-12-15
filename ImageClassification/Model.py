import tensorflow as tf
import tensorflow_hub as hub

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._top_k = params['TOP_K']
        self._prepare_graph(params)

    def _prepare_graph(self, params):
        tf.reset_default_graph()
        self.dataset = Dataset(params)

        logits = self._prepare_model(self.dataset.img_data)
        softmax = tf.nn.softmax(logits)
        self.top_prediction = tf.nn.top_k(softmax, self._top_k, name='top_prediction')

    def _prepare_model(self, images):
        module = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/classification/1')
        logits = module(dict(images=images))
        return logits
