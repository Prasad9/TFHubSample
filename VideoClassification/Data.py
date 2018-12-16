import os
import pandas as pd
import csv

class Data:
    def __init__(self, params):
        self._data_path = os.path.abspath(params['DATA_PATH'])
        self._train_labels = os.path.abspath(params['TRAIN_LABELS'])
        self._val_labels = os.path.abspath(params['VAL_LABELS'])
        self._test_labels = os.path.abspath(params['TEST_LABELS'])
        self._reqd_labels = params['REQD_LABELS']

        self._label_index_map, self._index_label_map = self._fetch_labels(os.path.abspath(params['LABEL_TEXT']))
        print(self._label_index_map, self._index_label_map)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.X_test_filenames = self._prepare_data()

    def _fetch_labels(self, label_text):
        label_index_map = {}
        index_label_map = {}
        with open(label_text, 'r') as fid:
            csv_fid = csv.reader(fid)
            current_index = 0
            for line_id, label in enumerate(csv_fid):
                if label[0] in self._reqd_labels:
                    label_index_map[label[0]] = current_index
                    index_label_map[current_index] = label[0]
                    current_index += 1

        return label_index_map, index_label_map

    def _prepare_data(self):
        data_pd = pd.read_csv(self._train_labels, names=['videoname', 'class_id'], sep=';')
        X_train, y_train = data_pd['videoname'], data_pd['class_id']
        X_train = [os.path.join(self._data_path, str(fname)) for fname in X_train]
        y_train = [self._label_index_map[y] for y in y_train]

        data_pd = pd.read_csv(self._val_labels, names=['videoname', 'class_id'], sep=';')
        X_val, y_val = data_pd['videoname'], data_pd['class_id']
        X_val = [os.path.join(self._data_path, str(fname)) for fname in X_val]
        y_val = [self._label_index_map[y] for y in y_val]

        data_pd = pd.read_csv(self._test_labels, names=['videoname'])
        X_test_filenames = data_pd['videoname']
        X_test = [os.path.join(self._data_path, str(fname)) for fname in X_test_filenames]

        print(len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(X_test_filenames))

        return X_train, y_train, X_val, y_val, X_test, X_test_filenames

    def label_at_index(self, index):
        return self._index_label_map[index]

    def get_train_data_length(self):
        return len(self.y_train)

    def get_val_data_length(self):
        return len(self.y_val)

    def get_test_data_length(self):
        return len(self.X_test)
