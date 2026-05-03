from code.base_class.setting import setting

class Setting_Train_Test_File(setting):
    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        self.method.data = loaded_data
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.fold_count = None
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None