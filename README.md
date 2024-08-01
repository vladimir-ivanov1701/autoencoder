# Автоэнкодер

Проект нейросети для очистки документов от шумов.

## 1. Первоначальная настройка

### 1.1. Клонирование репозитория
```
git clone https://github.com/vladimir-ivanov1701/autoencoder.git
cd autoencoder
```

### 1.2. Установка виртуального окружения и библиотек
```
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

**ВАЖНО!** Работосопособность проверялась на Python версии 3.7.9. На других версиях возможны конфликты библотек.

## 2. Запуск парсера
**ВАЖНО!** Код выложен в ознакомительных целях, это не полный проект. Парсер работать не будет.

### 2.1. Указание путей к файлам
В файле **constants.py** необходимо изменить пути к файлам на корректные.

### 2.2. Запуск парсера
```
python dev/pdf_parser/run_parser.py
```

## 3. Запуск обучения нейросети.

Для обучения с нуля необходимо убедиться, что в файле **constants.py** значение переменной **USE_PRETRAINED_MODEL** указано как False. Если необходимо дообучить ранее обученную модель, нужно указать значение True и в переменной **PATH_WEIGHTS** указать путь к сохранённым весам модели.

Для запуска обучения необходимо выполнить скрипт **autoencoder.py**:
```
python dev/autoencoder/autoencoder.py
```
