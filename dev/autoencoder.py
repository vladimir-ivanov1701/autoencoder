import os

import matplotlib.pyplot as plt
import torch
from constants import (CROP_SIZE, DIR_IMAGES_CLEANED, DIR_IMAGES_DIRTY,
                       DIR_REFERENCES_DIRTY, DUMP_PATH, GRAPH_PATH, IMG_HEIGHT,
                       IMG_WIDTH, N_EPOCHS, PATH_WEIGHTS, RANDOM_STATE,
                       TEST_SIZE, USE_PRETRAINED_MODEL)
from functions import show_pair_img
from modules import Autoencoder, MyModel, NoisyCleanDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms

# Формирование обучающей и тестовой выборки.
train_imgs, test_imgs = train_test_split(
    os.listdir(DIR_IMAGES_DIRTY),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

device = "cuda: 0" if torch.cuda_is_available() else "cpu"
MyModel.device = device
print(f"Using {device}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor
])

'''
Инициализация автоэнкодера.
Если нужно дообучить модель - в constants поставить:
    - USE_PRETRAINED_MODEL = True
    - DUMP_PATH = *путь к сохранённым весам*
'''

AutoEncoder = MyModel(NoisyCleanDataset, Autoencoder, transform)
if USE_PRETRAINED_MODEL:
    AutoEncoder.load_weights(PATH_WEIGHTS)

AutoEncoder.fit(
    N_EPOCHS,
    DIR_IMAGES_DIRTY,
    DIR_IMAGES_CLEANED,
    train_imgs,
    test_imgs
)

AutoEncoder.save_weights(DUMP_PATH)
AutoEncoder.predict(DIR_REFERENCES_DIRTY)

# сохранение изображения со сравнением исходной и очищенной картинок
img_list = os.listdir(DIR_REFERENCES_DIRTY)
for i in img_list:
    show_pair_img(i, f"outputs/outputs_{N_EPOCHS}_epochs")

# график кривой убывания лосса
f, ax = plt.subplots(figsize=(10, 10))
ax.plot(AutoEncoder.train_loss, color="red", label="train")
ax.plot(AutoEncoder.val_los, color="green", label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.savefig(GRAPH_PATH)
