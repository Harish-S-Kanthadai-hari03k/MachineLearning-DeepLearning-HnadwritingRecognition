import os
from datetime import datetime
from your_project.configs import BaseModelConfigs

class HandwritingRecognitionModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/03_handwriting_recognition", datetime.now().strftime("%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20

# Usage:
# config = HandwritingRecognitionModelConfigs()
# print(config.model_path)
