import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from constants import (DIR_REFERENCES_CLEANED, DIR_REFERENCES_DIRTY,
                       IMG_HEIGHT, IMG_WIDTH, N_EPOCHS)
from torchvision.transforms import transforms


def show_pair_img(img, img_path, n_epochs=N_EPOCHS):
    '''
    Функция объединяет на одном изображении три скана:
        - исходный "грязный"
        - очищенный
        - эталонный чистый

    Это необходимо для возможности визуального контроля того, как
    обучается модель и как меняется качество очистки изображений.
    '''

    os.makedirs(f"outputs/outputs_{n_epochs}_epochs_compared", exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_HEIGHT, IMG_WIDTH)
    ])

    img_noisy = data_transforms(
        cv2.imread(f"{DIR_REFERENCES_DIRTY}/{img}")
    )
    img_clean = data_transforms(
        cv2.imread(f"{DIR_REFERENCES_CLEANED}/{img}")
    )
    img_cleaned = mpimg.imread(f"{img_path}/{img}")

    titles = ["Clean", "Noisy", "Cleaned"]
    images = [img_clean, img_noisy, img_cleaned]

    for i in range(3):
        ax[i].imshow(images[i], cmap="gray")
        ax[i].axis["off"]
        ax[i].set_title(titles[i], fontsize=20)

    plt.savefig(f"{img_path}_compared/{img}")
    plt.cla()
