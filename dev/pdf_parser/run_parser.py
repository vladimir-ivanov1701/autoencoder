import warnings

warnings.filterwarnings("ignore")

import logging
import multiprocessing
import os
import shutil
import sys

from constants import (BASE_DIR, DS_DIR, HOME_DIR, MAX_PROCESSES, PATH_CLEANED,
                       PATH_DIRTY)

# настройка окружения - нужно для корректной работы скрипта
os.environ["TESSDATA_PREFIX"] = f"/home/user/all_files/{BASE_DIR}/var/tessdata"
os.environ["PATH"] = "/tmp/venv/bin/"
os.environ["MULTIPROCESSING_CONTEXT"] = "fork"
os.chdir("/home/user/autoencoder")

if HOME_DIR not in sys.path:
    sys.path.append(HOME_DIR)

from logging import Logger
from multiprocessing import get_context, log_to_stderr

from process_pdf import initialize, process_one_page, push_to_s3
from src.pdf import get_pdf_pages_number
from src.pipelines import setup_templates, setup_universal_parser

MULTIPROCESSING_CONTEXT = os.environ.get("MULTIPROCESSING_CONTEXT", "spawn")
IMAGES_PREPROCESSORS_NUMBER = int(os.environ.get("IMAGES_PREPROCESSORS_NUMBER", 4))

logger = Logger("test_OCR")
templates = setup_templates(drop_cache=False)
usp = setup_universal_parser(templates=templates, logger=logger)

# собираем список файлов, которые нужно обработать
files_list = []

for filename in os.listdir(BASE_DIR):
    f = os.path.join(BASE_DIR, filename)
    files_list.append(f)

'''
Итеративно проходимся по каждому файлу. Для экономии места на диске
каждые 10 файлов выталкиваем полученные изображения в S3 и очищаем
место на диске.
'''
for x in range(0, len(files_list)):
    if x != 0 and x % 10 == 0:
        for d in (PATH_DIRTY, PATH_CLEANED, BASE_DIR):
            jobs = [(d, f) for f in os.listdir(d)]
            pool = multiprocessing.Pool(MAX_PROCESSES, initialize)
            pool.map(push_to_s3, jobs)
            pool.close()
            pool.join()

        # удаляем файлы после загрузки
        for d in (PATH_DIRTY, PATH_CLEANED):
            try:
                shutil.rmtree(d)
            except:
                os.makedirs(d, exist_ok=True)

    f = files_list[x]
    f_name = f.split("/")[-1][:-4]

    n_pages = get_pdf_pages_number(f)
    for p_num in range(0, n_pages):
        process_one_page(
            args=[
                f,
                p_num,
                usp._tesseract_conf,
                usp._noise_regressor,
                logger,
                "pdf",
                f_name
            ]
        )

print("All files processed!")
