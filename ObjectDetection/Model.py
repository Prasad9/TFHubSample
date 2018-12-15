import tensorflow as tf
import tensorflow_hub as hub

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._prepare_graph(params)

    def _prepare_graph(self, params):
        tf.reset_default_graph()
        self.dataset = Dataset(params)

        self.class_entities, self.boxes = self._prepare_model(self.dataset.img_data)

    def _prepare_model(self, images):
        module = hub.Module('https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
        detector = module(dict(images=images), as_dict=True)

        class_entities = detector['detection_class_entities']
        boxes = detector['detection_boxes']

        return class_entities, boxes
