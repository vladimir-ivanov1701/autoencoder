# пути к папкам с изображениями
DIR_IMAGES = "path_to_dataset"
DIR_IMAGES_DIRTY = DIR_IMAGES + "/dirty"
DIR_IMAGES_CLEANED = DIR_IMAGES + "/cleaned"
DIR_REFERENCES = DIR_IMAGES + "/references"
DIR_REFERENCES_DIRTY = DIR_REFERENCES + "/dirty"
DIR_REFERENCES_CLEANED = DIR_REFERENCES + "/cleaned"

# путь сохранения графика прогресса обучения
GRAPH_PATH = "path_to_graph/graph.png"

# параметры обучения
N_EPOCHS = 50
RANDOM_STATE = 42
TEST_SIZE = 0.33
USE_PRETRAINED_MODEL = False

# параметры изображений
CROP_SIZE = 900
IMG_HEIGHT = 880
IMG_WIDTH = 624

# путь сохранения модели
DUMP_PATH = f"weights_folder/model_{N_EPOCHS}_weights.pth"
PATH_WEIGHTS = "weights_folder/model_weights_to_load.pth"
