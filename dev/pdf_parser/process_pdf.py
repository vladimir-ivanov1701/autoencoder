from PIL import Image, ImageFilter
from src.image import LayerImage
from src.utils import (
    get_images_from_pdf_page,
    get_images_from_tiff,
    tesseract_OSD
)

from constants import (
    BUCKET_NAME,
    DESTINATION,
    PATH_CLEANED,
    PATH_DIRTY,
    URL
)

from typing import Tuple, List

import os
import boto3

os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("JUPYTERHUB_USER")

# здесь нужно указать любую случайную строку
os.environ["AWS_SECRET_ACCESS_KEY"] = "123qq"


def change_contrast(img: Image, level: int) -> Image:
    '''Изменение контрастности изображения.'''

    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def change_brightness(img: Image, lvl: int) -> Image:
    res = img.point(lambda x: x + lvl)
    return res


def initialize():
    '''Инициализация клиента S3.'''
    global s3_client
    s3_client = boto3.client(service_name="s3", endpoint_url=URL, verify=False)


def push_to_s3(job: Tuple):
    '''
    Проверяет наличие файла в S3, если не найден - выталкивает его туда.
    '''

    path_dir, f_name = job
    local_path = f"{path_dir}/{f_name}"
    if os.path.isfile(local_path):
        dir_name = path_dir.split("/")[-1]
        s3_path = f"{DESTINATION}/{dir_name}/{f_name}"
        print('Searching "%s" in "%s"' % (s3_path, BUCKET_NAME))
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_path)
        except:
            print("Uploading %s..." % s3_path)
            s3_client.upload_file(local_path, BUCKET_NAME, s3_path)


def process_one_page(args: List) -> dict:
    '''Обработка одной страницы исходного файла.'''

    bytestream = args[0]
    page_num = args[1]
    tesseract_config = args[2]
    noise_regressor = args[3]
    logger = args[4]
    source_type = args[5]
    f_name = args[6]

    layer_image = None

    if source_type == "pdf":
        get_data_func = get_images_from_pdf_page
    elif "tiff" == source_type:
        get_data_func = get_images_from_tiff
    else:
        logger.critical(
            f"Can't choose data extraction method for source {type(bytestream)}"
        )

    images = [
        LayerImage(
            image_id=im_id,
            image=image[0],
            noise_regressor=noise_regressor,
            logger=logger,
        )
        for im_id, image in get_data_func(bytestream, page_num, logger).items()
        if image
    ]

    if images:
        layer_image = images[0]
        img_dirty = layer_image.image
        layer_image = tesseract_OSD(((layer_image,), tesseract_config))
        layer_image.preprocess()
        img_cleaned = layer_image.image

        # изменения яркости и контрастности
        for m in range(-100, 101, 100):
            img = change_contrast(img_dirty, m)
            for n in range(-100, 101, 100):
                img = change_brightness(img, n)
                img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
                for d in [PATH_DIRTY, PATH_CLEANED]:
                    os.makedirs(d, exist_ok=True)
                img.save(
                    f"{PATH_DIRTY}/{f_name}_CON_{m}_BR_{n}_{page_num}.png"
                )
                img_cleaned.save(
                    f"{PATH_CLEANED}/{f_name}_CON_{m}_BR_{n}_{page_num}.png"
                )
    return None
