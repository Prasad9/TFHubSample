import cv2
import tensorflow as tf
from collections import deque
import os
import numpy as np

class Infer:
    def __init__(self, params):
        self._seq_len = params['SEQ_LEN']
        self._img_width = params['IMG_WIDTH']
        self._img_height = params['IMG_HEIGHT']

        self._labels = params['LABELS']
        self._n_class = len(self._labels)
        self._checkpoint_path = params['CHECKPOINT_PATH']

        self._img_frames = deque(maxlen=self._seq_len)
        self._camera = cv2.VideoCapture(0)
        self._create_graph()

    def _create_graph(self):
        self._sess = tf.Session()
        self._sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        tf.saved_model.loader.load(self._sess, [tf.saved_model.tag_constants.SERVING], self._checkpoint_path)

        logits_op = self._sess.graph.get_tensor_by_name('logits:0')
        self._predictions_op = tf.argmax(logits_op, axis=1)
        self._data_X_op = self._sess.graph.get_tensor_by_name('data_X:0')

    def _process_image(self, img):
        img = cv2.resize(img, (self._img_width, self._img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Our sequence length (36) of images in one video is huge and fps of camera is quite low.
        # To fill the buffer quickly, fill three images.
        self._img_frames.append(img)
        self._img_frames.append(img)
        self._img_frames.append(img)

        if len(self._img_frames) == self._seq_len:
            img_array = np.array(self._img_frames, dtype=np.float32)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = self._sess.run(self._predictions_op, feed_dict={self._data_X_op: img_array})[0]
            prediction = self._labels[prediction]
            return prediction

        return None

    def infer_camera(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            ret, frame = self._camera.read()

            info = self._process_image(frame)
            if info is not None:
                cv2.putText(frame, info, (220, 440), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Video Classification example', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self._camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    params = {
        'SEQ_LEN': 36,
        'IMG_WIDTH': 224,
        'IMG_HEIGHT': 224,

        # List all the labels you have trained your model on. Do not change the order.
        'LABELS': ['Swiping Left', 'Swiping Right', 'Swiping Down',
                    'Swiping Up', 'Pushing Hand Away', 'Pulling Hand In',
                    'Sliding Two Fingers Left', 'Sliding Two Fingers Right',
                    'Sliding Two Fingers Down', 'Sliding Two Fingers Up',
                    'Pushing Two Fingers Away', 'Pulling Two Fingers In',
                    'Rolling Hand Forward', 'Rolling Hand Backward', 'Turning Hand Clockwise',
                    'Turning Hand Counterclockwise', 'Zooming In With Full Hand',
                    'Zooming Out With Full Hand', 'Zooming In With Two Fingers',
                    'Zooming Out With Two Fingers', 'Thumb Up', 'Thumb Down',
                    'Shaking Hand', 'Stop Sign', 'Drumming Fingers',
                    'No gesture', 'Doing other things'],

        'CHECKPOINT_PATH': os.path.abspath('./Model/3')   # Path where your saved_model is present
    }
    i = Infer(params)
    i.infer_camera()