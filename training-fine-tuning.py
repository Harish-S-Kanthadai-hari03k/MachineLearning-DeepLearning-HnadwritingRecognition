import tensorflow as tf
try:
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from your_project.preprocessors import ImageReader
from your_project.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from your_project.augmentors import CustomBrightness, CustomRotate, CustomErodeDilate, CustomSharpen
from your_project.annotations.images import CustomCVImage
from your_project.tensorflow.data_provider import CustomDataProvider
from your_project.tensorflow.losses import CustomCTCLoss
from your_project.tensorflow.callbacks import CustomModel2onnx, CustomTrainLogger
from your_project.tensorflow.metrics import CustomCWERMetric
from your_project.models import custom_train_model
from your_project.configs import CustomModelConfigs

import os
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

def download_and_extract(url, extract_to="CustomDatasets", chunk_size=1024 * 1024):
    http_response = urlopen(url)

    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    zipfile = ZipFile(BytesIO(data))
    zipfile.extractall(path=extract_to)

dataset_path = os.path.join("CustomDatasets", "IAM_Words")
if not os.path.exists(dataset_path):
    download_and_extract("https://git.io/J0fjL", extract_to="CustomDatasets")

    file = tarfile.open(os.path.join(dataset_path, "words.tgz"))
    file.extractall(os.path.join(dataset_path, "words"))

dataset, vocab, max_len = [], set(), 0

# Preprocess the dataset by the specific IAM_Words dataset file structure
words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[1] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = "-".join(line_split[0].split("-")[:2])
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n")

    rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
    if not os.path.exists(rel_path):
        print(f"File not found: {rel_path}")
        continue

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Create a CustomModelConfigs object to store model configurations
configs = CustomModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = CustomDataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CustomCVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Augment training data with custom brightness, rotation, and erode/dilate
train_data_provider.augmentors = [
    CustomBrightness(),
    CustomErodeDilate(),
    CustomSharpen(),
    CustomRotate(angle=10),
]

# Creating a custom TensorFlow model architecture
model = custom_train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CustomCTCLoss(),
    metrics=[CustomCWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/custom_model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = CustomTrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto")
model2onnx = CustomModel2onnx(f"{configs.model_path}/custom_model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save training and validation datasets as CSV files
train_data_provider.to_csv(os.path.join(configs.model_path, "custom_train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "custom_val.csv"))
