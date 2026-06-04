from code.base_class.result import result
import pickle


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        with open(
            self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count),
            'wb',
        ) as f:
            pickle.dump(self.data, f)

    def load(self):
        with open(
            self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count),
            'rb',
        ) as f:
            return pickle.load(f)
