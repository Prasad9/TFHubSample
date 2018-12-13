import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.contrib.image

from Dataset import Dataset

class Model:
    def __init__(self, params):
        self._image_size = params['OUTPUT_IMAGE_SIZE']

        self._build_architecture()

    def _build_architecture(self):
        tf.reset_default_graph()

        self.dataset = Dataset()
        self.aug_images = self._build_model()

    def _build_model(self):
        module = hub.Module('https://tfhub.dev/google/image_augmentation/nas_svhn/1')
        #print(module.get_signature_names())
        #print(module.get_input_info_dict(signature='from_decoded_images'))
        input_dict = dict(image_size=self._image_size,
                          images=self.dataset.img_data,
                          augmentation=True)
        aug_images = module(input_dict, signature='from_decoded_images')
        return aug_images