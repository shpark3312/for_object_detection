import math
import os
from collections import defaultdict
import cv2
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split

def crop(img, patch_width=1280, patch_height=720, x_overlay=80, y_overlay=45):

    result = []
    x_step = patch_width - x_overlay
    y_step = patch_height - y_overlay
    height, width, _ = img.shape

    for x_offset in range(math.floor((width - patch_width) / x_step) + 1):
        for y_offset in range(math.floor((height - patch_height) / y_step) + 1):
            xstart = max(0, x_step * x_offset)
            ystart = max(0, y_step * y_offset)
            xend = min(width, xstart + patch_width)
            yend = min(height, ystart + patch_height)
            result.append(
                {
                    "img": img[ystart:yend, xstart:xend],
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "xstart": xstart,
                    "ystart": ystart,
                    "xend": xend,
                    "yend": yend,
                }
            )

    return result


def parse_GTcsv(csv_path):
    label_dict = defaultdict(lambda: {"labels": [], "bboxes": []})

    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for line in rdr:
            if len(line) < 6:
                continue
            if line[1] == "":
                continue
            filename = line[0]
            bbox = list(round(int(i)) for i in line[1:5])
            category = line[-1]
            label_dict[filename]["labels"].append(category)
            label_dict[filename]["bboxes"].append(bbox)

    return label_dict


def crop_images(names, OUT_IMG_DIR, GT_data, f):

    for name in names:

        label_list = GT_data[name]["labels"]
        box_list = GT_data[name]["bboxes"]

        img = cv2.imread(os.path.join(os.path.dirname(GT_CSV_PATH), name))

        pure_filename = os.path.splitext(os.path.split(name)[-1])[0]
        if np.all(img) == None:
            continue

        cropped_patches = crop(img, patch_width, patch_height, x_overlay, y_overlay)

        for cropped_patch in cropped_patches:
            patch = cropped_patch["img"]
            x_offset = cropped_patch["x_offset"]
            y_offset = cropped_patch["y_offset"]
            xstart = cropped_patch["xstart"]
            ystart = cropped_patch["ystart"]
            xend = cropped_patch["xend"]
            yend = cropped_patch["yend"]

            out_img_dir = os.path.join(
                f"{OUT_IMG_DIR}", f"{pure_filename}_{x_offset}_{y_offset}.jpg"
            )

            cv2.imwrite(os.path.join(os.path.dirname(GT_CSV_PATH), out_img_dir), patch)

            for b, l in zip(box_list, label_list):
                x1, y1, x2, y2 = [*b]
                b = [x1 - xstart, y1 - ystart, x2 - xstart, y2 - ystart]
                x_pad = 40
                y_pad = 70
                if (
                    (xstart - x_pad < x1)
                    and (x2 < xend + x_pad)
                    and (ystart - y_pad < y1)
                    and (y2 < yend + y_pad)
                ):
                    if (xstart - x_pad <= x1) and (x1 <= xstart):
                        b[0] = 0
                    if (xend <= x2) and (x2 <= xend + x_pad):
                        b[2] = patch_width
                    if (ystart - y_pad <= y1) and (y1 <= ystart):
                        b[1] = 0
                    if (yend <= y2) and (y2 <= yend + y_pad):
                        b[3] = patch_height

                    f.write("{},{},{},{},{},{}\n".format(out_img_dir, *b, l))

    f.close()

# def make_cropped_dataset(GT_data, GT_CSV_PATH, patch_width, patch_height, x_overlay, y_overlay):



if __name__ == "__main__":



    ##################################여기 수정 ##################################
    GT_CSV_PATH_LIST = ["dataset/parsed_dataset/2021_04_15_DJI_3_fps.csv",
                        "dataset/parsed_dataset/20210518_02_7_fps.csv",
                        "dataset/parsed_dataset/20210525_EO_Movie_7_fps.csv",
                        "dataset/parsed_dataset/20210526_EO_Movie_7_fps.csv",
                        "dataset/parsed_dataset/20210601_EO_Movie_7_fps.csv",
                        "dataset/parsed_dataset/20210603_EO_Movie_7_fps.csv",
                        ]
    ######################################################################################################


    patch_width = 832
    patch_height = 832
    x_overlay = 80
    y_overlay = 168

    for GT_CSV_PATH in GT_CSV_PATH_LIST:
        OUT_TRAIN_CSV_PATH = f'{GT_CSV_PATH.split(".")[0]}_cropped_train.csv'
        OUT_VAL_CSV_PATH = f'{GT_CSV_PATH.split(".")[0]}_cropped_val.csv'
        OUT_IMG_DIR = f'{os.path.splitext(GT_CSV_PATH)[0]}_cropped'

        RELITIVE_OUT_IMG_DIR = f'{os.path.split(GT_CSV_PATH)[-1].split(".")[0]}_cropped'

        f_train = open(os.path.join(OUT_TRAIN_CSV_PATH), "w")
        f_val = open(os.path.join(OUT_VAL_CSV_PATH), "w")

        os.makedirs(OUT_IMG_DIR, exist_ok=True)
        GT_data = parse_GTcsv(GT_CSV_PATH)
        file_list = list(GT_data.keys())

        train_names, val_names = train_test_split(file_list, test_size = 0.2, random_state = 1)

        crop_images(train_names, RELITIVE_OUT_IMG_DIR, GT_data, f_train)
        crop_images(val_names, RELITIVE_OUT_IMG_DIR, GT_data, f_val)
