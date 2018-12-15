import os

class Data:
    def __init__(self, params):
        infer_path = os.path.abspath(params['INFER_PATH'])

        self.infer_files = os.listdir(infer_path)
        self.infer_files = [os.path.join(infer_path, p) for p in self.infer_files]

    def get_infer_data_length(self):
        return len(self.infer_files)