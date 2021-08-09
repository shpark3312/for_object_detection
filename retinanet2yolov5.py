import csv
import os
import cv2
import imagesize
from shutil import copyfile
from collections import defaultdict

def read_text_file_by_line(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for i in f:
            i = i.strip()
            if i:
                yield i


def parse_categories_file(csv_path):
    result = {}
    for line in read_text_file_by_line(csv_path):
        split = line.split(",")
        if len(split) < 2 or any(not i for i in split):
            continue
        result[split[0]] = int(split[1])
    return result


def parse_csv_file(IN_CSV_PATH):
    data_dict = defaultdict(lambda: {'bbox':[], 'label':[]})
    with open(IN_CSV_PATH, "r", encoding="UTF8") as f:
            rdr = csv.reader(f)
            for line in rdr:
                print(line)
                if line[5] == "":
                    continue
                else:
                    data_dict[line[0]]['bbox'].append([line[1], line[2], line[3], line[4]])
                    data_dict[line[0]]['label'].append(line[5])

    print(data_dict)

    return 0




def csv2yolo(IN_CSV_PATH, basepath, task):
    if IN_CSV_PATH:
        with open(IN_CSV_PATH, "r", encoding="UTF8") as f:
            rdr = csv.reader(f)
            # total_lines = sum(1 for row in rdr)
            # rdr = csv.reader(f)
            # print(rdr.line_num)
            for line in rdr:
                print(line)
                if line[5] == "":
                    continue
                else:
                    filename = line[0]
                    file_dir_list = os.path.split(filename)
                    pure_filename = file_dir_list[-1].split(".")[0]
                    image_width, image_height = imagesize.get(
                        os.path.join(os.path.dirname(IN_CSV_PATH), filename)
                    )
                    copyfile(
                        os.path.join(os.path.dirname(IN_CSV_PATH), filename),
                        os.path.join(
                            basepath,
                            "images",
                            task,
                            f"{line[0].split('/')[0]}_{line[0].split('/')[-1].split('.')[0]}.jpg",
                        ),
                    )

                    x1 = round(float(line[1]))
                    y1 = round(float(line[2]))
                    x2 = round(float(line[3]))
                    y2 = round(float(line[4]))
                    cat_num = CATEGORY_TABLE[line[5]]
                    center_x = ((x1 + x2) / 2) / image_width
                    center_y = ((y1 + y2) / 2) / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height

                    with open(
                        os.path.join(
                            basepath,
                            "labels",
                            task,
                            f"{line[0].split('/')[0]}_{line[0].split('/')[-1].split('.')[0]}.txt",
                        ),
                        "a",
                    ) as t:
                        data = f"{cat_num} {center_x} {center_y} {width} {height}\n"
                        t.write(data)


if __name__ == "__main__":

    ############################여기 수정############################
    IN_PATH = "./dataset"
    IN_TRAIN_CSV_PATH_LIST = [
        "./dataset/parsed_dataset/01_cropped/20210513_EO_Movie_7_fps_cropped_train.csv",
        "./dataset/parsed_dataset/01_cropped/20210518_01_7_fps_cropped_train.csv",
        "./dataset/parsed_dataset/01_cropped/20210518_01_30_fps_cropped_train.csv",
    ]

    IN_VAL_CSV_PATH_LIST = [
        "./dataset/parsed_dataset/01_cropped/20210513_EO_Movie_7_fps_cropped_val.csv",
        "./dataset/parsed_dataset/01_cropped/20210518_01_7_fps_cropped_val.csv",
        "./dataset/parsed_dataset/01_cropped/20210518_01_30_fps_cropped_val.csv",
    ]


    basepath = "./dataset/for_yolo/20210624"
    CATEGORY_PATH = "./dataset/classes.csv"

    ################################################################

    CATEGORY_TABLE = parse_categories_file(CATEGORY_PATH)
    REVERSE_CATEGORY_TABLE = {
        v: k for k, v in CATEGORY_TABLE.items()
    }  # indices -> name



    parse_csv_file(IN_TRAIN_CSV_PATH_LIST[0])

    # if not os.path.exists(basepath):
    #     os.mkdir(basepath)
    #     os.mkdir(os.path.join(basepath, "images"))
    #     os.mkdir(os.path.join(basepath, "images", "train"))
    #     os.mkdir(os.path.join(basepath, "images", "val"))
    #     os.mkdir(os.path.join(basepath, "labels"))
    #     os.mkdir(os.path.join(basepath, "labels", "train"))
    #     os.mkdir(os.path.join(basepath, "labels", "val"))

    # for IN_TRAIN_CSV_PATH in IN_TRAIN_CSV_PATH_LIST:
    #     csv2yolo(IN_TRAIN_CSV_PATH, basepath, "train")

    # for IN_VAL_CSV_PATH in IN_VAL_CSV_PATH_LIST:
    #     csv2yolo(IN_VAL_CSV_PATH, basepath, "val")
