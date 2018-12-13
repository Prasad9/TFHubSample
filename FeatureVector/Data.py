import os
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, params):
        self._train_path = os.path.abspath(params['TRAIN_PATH'])
        self._test_path = os.path.abspath(params['TEST_PATH'])
        self._train_val_ratio = params['TRAIN_VAL_RATIO']
        self._labels = os.path.abspath(params['DATA_LABELS'])

        self.X_train, self.y_train, self.X_val, self.y_val = self._prepare_train_val_data()
        self.X_test = self._prepare_test_data()

    def _prepare_train_val_data(self):
        data_pd = pd.read_csv(self._labels)
        X_data, y_data = data_pd['Image_name'], data_pd.iloc[:, 1:]
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, random_state=198,
                                                          train_size=self._train_val_ratio)
        X_train = [os.path.join(self._train_path, p) for p in X_train]
        X_val = [os.path.join(self._train_path, p) for p in X_val]
        return X_train, y_train, X_val, y_val

    def _prepare_test_data(self):
        test_files_count = len(os.listdir(self._test_path))
        # Submission file is accepted in a serial order only.
        test_files = [os.path.join(self._test_path, 'Image-{}.jpg'.format(i + 1))
                        for i in range(test_files_count)]
        return test_files

    def get_train_data_length(self):
        return len(self.y_train)

    def get_val_data_length(self):
        return len(self.y_val)

    def get_test_data_length(self):
        return len(self.X_test)