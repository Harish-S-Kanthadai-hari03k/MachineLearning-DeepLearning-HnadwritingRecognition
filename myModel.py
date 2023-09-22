import cv2
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
from your_project.inferenceModel import OnnxInferenceModel
from your_project.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(model_path, *args, **kwargs)

    def predict(self, image: np.ndarray) -> str:
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.vocab)[0]
        return text

if __name__ == "__main__":
    model_path = "Models/08_handwriting_recognition_tf/202303142139/model.onnx"
    val_csv_path = "Models/08_handwriting_recognition_tf/202303142139/val.csv"

    model = ImageToWordModel(model_path=model_path)
    df = pd.read_csv(val_csv_path).values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")
