import tensorflow as tf
import tensorflow_hub as hub

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._build_graph(params)

    def _build_graph(self, params):
        tf.reset_default_graph()
        self.dataset = Dataset(params)

        module = hub.Module("https://tfhub.dev/google/progan-128/1")
        self.images = module(self.dataset.latent_vector_space)
