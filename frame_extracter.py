import math
import os
import random
import time
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import cv2
import matplotlib
import numpy as np


def log(log_text: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {log_text}")


def get_filename_without_ext(path: str) -> str:
    fn, _ = os.path.splitext(os.path.basename(path))
    return fn


if __name__ == "__main__":


    ########################################################################################################################################
    seconds_per_image = 20

    # IN_VIDEOS_PATH = "dataset/201105_sortie/orig"
    IN_VIDEOS_PATH = r'D:\inspace\2021\DNA\dna_water\person_detection\dataset\202106_video'

    ########################################################################################################################################



    OUT_VIDEOS_PATH = f"{IN_VIDEOS_PATH}_{seconds_per_image}_fps"

    os.makedirs(OUT_VIDEOS_PATH, exist_ok=True)
    current_time = time.time()

    for root, dirs, files in os.walk(IN_VIDEOS_PATH):
        for i, filename in enumerate(files):
            if os.path.splitext(filename)[-1].lower() not in [".mp4", ".mov"]:
                continue

            in_filepath = os.path.join(root, filename)
            out_filepath = os.path.join(OUT_VIDEOS_PATH, filename)

            print(f"{in_filepath} -> {out_filepath}")
            pure_filename = get_filename_without_ext(filename)
            log("처리 시작: {}".format(pure_filename))

            current_time = time.time()
            in_vid = cv2.VideoCapture(in_filepath)
            fps = in_vid.get(cv2.CAP_PROP_FPS)

            total_frame = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))

            interval = int(round(fps) * seconds_per_image)
            frame_list = range(0, total_frame, interval)

            for frame_num in frame_list:

                in_vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _, frame = in_vid.read()
                img_filename = frame_num

                cv2.imwrite(
                    os.path.join(
                        OUT_VIDEOS_PATH, f"{pure_filename}_{img_filename}.jpg"
                    ),
                    frame,
                )

                print(
                    f"Extracting... total video : {i}/{len(files)} : {math.floor(frame_num * 100 / total_frame)}%",
                    end="\r",
                    flush=True,
                )
