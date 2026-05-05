import json


class DivingConfig():

    def __init__(self, filename="../config.json"):
        with open(filename) as config_file:
            self.config = json.load(config_file)
    
    def getAll(self):
        return self.config

    def getInfo(self):
        return self.config.get("standard", {})

    def trainingConfig(self):
        return self.config.get("training", {})

    def getDataConfig(self):
        return self.config.get("dataset", {})


    def evalConfig(self):
        return self.config.get("evaluation", {})

    # Convenience getters for common hyperparameters and settings
    def get_name(self):
        return self.getInfo().get("name")

    def get_dataset(self):
        return self.getDataConfig()

    def get_training(self):
        return self.trainingConfig()

    def get_epochs(self):
        return int(self.get_training().get("epochs", 30))

    def get_batch_size(self):
        return int(self.get_training().get("batch_size", 8))

    def get_model_name(self):
        return self.get_training().get("model", None)

    def get_save_settings(self):
        train = self.get_training()
        return {
            "save_dir": train["save_dir"],
            "save_model": train["save_model"]
        }

    def get_labels(self):
        """Get label categories and label map for evaluation/visualization."""
        labels_cfg = self.config.get("labels", {})
        return {
            "categories": labels_cfg.get("categories", []),
            "label_map": labels_cfg.get("label_map", {}),
        }

    def get_categories(self):
        """Get list of diving categories."""
        return self.get_labels().get("categories", [])

    def get_label_map(self):
        """Get mapping from label indices to category names."""
        return self.get_labels().get("label_map", {})
