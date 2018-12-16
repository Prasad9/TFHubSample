import os
import pandas as pd

from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, params):
        self._train_file = os.path.abspath(params['TRAIN_FILE'])
        self._test_file = os.path.abspath(params['TEST_FILE'])
        self._train_val_split = params['TRAIN_VAL_RATIO']

        self._fetch_data()

    def _fetch_data(self):
        X_data, y_data = self._read_train_data()
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_data, y_data,
                                                                        train_size=self._train_val_split)

        self.X_test, self.X_test_id = self._read_test_data()

    def _read_train_data(self):
        pd_data = pd.read_csv(self._train_file, sep='\t')
        X_data, y_data = pd_data['Phrase'], pd_data['Sentiment']
        return X_data, y_data

    def _read_test_data(self):
        pd_data = pd.read_csv(self._test_file, sep='\t')
        X_data, X_data_id = pd_data['Phrase'], pd_data['PhraseId']
        return X_data, X_data_id

    def get_train_data_length(self):
        return len(self.y_train)

    def get_val_data_length(self):
        return len(self.y_val)

    def get_test_data_length(self):
        return len(self.X_test_id)
