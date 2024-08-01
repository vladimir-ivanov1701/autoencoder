import os
from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from constants import DIR_REFERENCES_DIRTY, IMG_HEIGHT, IMG_WIDTH, N_EPOCHS
from functions import show_pair_img
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NoisyCleanDataset(Dataset):
    '''Класс датасета'''
    def __init__(self, noisy_path, images, clean_path=None, transforms=None):
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.images = images
        self.transforms = transforms

    def __len__(self) -> int:
        '''Длина датасета'''

        return len(self.images)

    def __getitem__(self, i) -> Tuple:
        '''
        Функция возвращает зашумлённое изображение и его имя в папке.
        Если указан путь до папки с чистыми изображениями - также
        возвращается чистое изображение.
        '''

        noisy_image = cv2.imread(f"{self.clean_path}/{self.images[i]}")
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

        if self.transforms:
            noisy_image = self.transforms(noisy_image)

        if self.clean_path:
            clean_image = cv2.imread(f"{self.clean_path}/{self.images[i]}")
            clean_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
            if self.transforms:
                clean_image = self.transforms(clean_image)
            return (noisy_image, clean_image, self.images[i])
        else:
            return (noisy_image, self.images[i])


class Autoencoder(nn.Module):
    '''
    Класс автоэнкодера. Состоит из 2 частей: энкодера и декодера.
    '''

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=1.5),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = nn.functional.interpolate(encoded, scale_factor=2)
        decoded = self.decoder(decoded)


class MyModel():
    '''Класс модели.'''

    def __init__(self, Dataset, Model, transforms):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.Dataset = Dataset
        self.model = Model().to(self.device)
        self.transform = transforms

    def load_weights(self, path):
        '''
        Загрузка сохранённых весов модели.
        '''

        self.model.load_state_dict(
            torch.load(
                path, map_location=(
                    torch.device("cpu") if self.device == "cpu" else None
                )
            )
        )

    def save_weights(self, path):
        '''
        Сохранение весов модели.
        '''

        torch.save(self.model.state_dict(), path)

    def fit(self, n_epochs, noisy_path, clean_path, train_imgs, test_imgs):
        '''Обучение модели.'''

        train_data = self.Dataset(
            noisy_path,
            train_imgs,
            clean_path,
            self.transform
        )

        val_data = self.Dataset(
            noisy_path,
            test_imgs,
            clean_path,
            self.transform
        )

        trainloader = DataLoader(
            train_data,
            batch_size=4,
            shuffle=False
        )

        valloader = DataLoader(
            val_data,
            batch_size=4,
            shuffle=False
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        self.train_loss = []
        self.val_loss = []
        running_loss = 0.0

        for epoch in range(n_epochs):
            self.model.train()
            for i, data in enumerate(trainloader):
                noisy_img = data[0].to(self.device)
                clean_img = data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(noisy_img)
                loss = criterion(outputs, clean_img)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # вывод промежуточной статистики каждые 10 батчей
                if i % 10 == 0:
                    print(f"Epoch {epoch + 1} batch {i}: Loss {loss.item()/4}")
            self.train_loss.append(running_loss / len(trainloader.dataset))

            '''Валидация'''

            print("Validation...")
            self.model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for i, data in tqdm(
                    enumerate(valloader),
                    total=int(len(val_data) / valloader.batch_size)
                ):
                    noisy_img = data[0].to(self.device)
                    clean_img = data[1].to(self.device)
                    outputs = self.model(noisy_img)
                    loss = criterion(outputs, clean_img)
                    running_loss += loss.item()
            current_val_loss = running_loss/len(valloader.dataset)
            self.val_loss.append(current_val_loss)
            print(f"Val loss: {current_val_loss:.5f}")

            '''Сохранение результатов каждые 10 эпох.'''

            if epoch % 10 == 0 and epoch > 0:
                self.save_weights(f"model_weights/model_{epoch}_epochs.pth")
                self.predict(DIR_REFERENCES_DIRTY, n_epochs=epoch)
                img_lst = os.listdir(DIR_REFERENCES_DIRTY)
                for i in img_lst:
                    show_pair_img(i, f"outputs/outputs_{epoch}_epochs", epoch)

    def save_predicted_img(self, dataset, dataloader, n_epochs):
        '''
        Функция принимает на вход датасет и даталоадер и
        сохраняет очищеннные моделью изображения.
        '''

        with torch.no_grad():
            for i, data in tqdm(
                enumerate(dataloader),
                total=int(dataset / dataloader.batch_size)
            ):
                noisy_img = data[0].to(self.device)
                outputs = self.model(noisy_img)
                transform_out = T.CenterCrop((IMG_HEIGHT, IMG_WIDTH))
                for im, im_name in zip(outputs, data[1]):
                    im = transform_out(im)
                    im = im.detach().cpu().permute(1, 2, 0).numpy()
                    cv2.imwrite(
                        f"outputs/outputs_{n_epochs}_epochs/{im_name}",
                        im*255
                    )

    def predict(self, img, n_epochs=N_EPOCHS):
        '''Предикт (очистка изображений моделью).'''

        os.makedirs(f"outputs/outputs_{n_epochs}_epochs", exist_ok=True)
        self.model.eval()
        if isinstance(img, str):
            if os.path.isfile(img):  # если предикт по одному файлу
                filename = os.path.basename(img)
                file_path = os.path.dirname(img)
                img = cv2.imread(img)
                predictDataset = self.Dataset(
                    file_path,
                    [filename],
                    transforms=self.transform
                )
                predictDataloader = DataLoader(
                    predictDataset,
                    batch_size=4,
                    shuffle=False
                )
            else:  # если предикт по всей папке
                images = os.listdir(img)
                predictDataset = self.Dataset(
                    img,
                    images,
                    transforms=self.transform
                )
                predictDataloader = DataLoader(
                    predictDataset,
                    batch_size=4,
                    shuffle=False
                )

            self.save_predicted_img(
                predictDataset,
                predictDataloader,
                n_epochs
            )
