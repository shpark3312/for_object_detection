import csv
import os
from collections import defaultdict



def parse_csv_file(IN_CSV_PATH):
    data_dict = defaultdict(lambda: {'bbox':[], 'label':[]})
    with open(IN_CSV_PATH, "r", encoding="UTF8") as f:
            rdr = csv.reader(f)
            for line in rdr:
                if line[5] == "":
                    continue
                else:
                    data_dict[line[0]]['bbox'].append([line[1], line[2], line[3], line[4]])
                    data_dict[line[0]]['label'].append(line[5])

    return dict(data_dict)


def data_dict_statictics(csv_dict):
    file_num = len(csv_dict)
    label_count = 0

    for filename in csv_dict.keys():
        label_count += len(csv_dict[filename]['bbox'])


    return file_num, label_count



if __name__ == '__main__':
    IN_CSV_LIST = ['./dataset/parsed_dataset/01_cropped/2021_04_15_DJI_3_fps_cropped_train.csv',
                    './dataset/parsed_dataset/01_cropped/2021_04_15_DJI_3_fps_cropped_val.csv',
                    './dataset/parsed_dataset/01_cropped/2021_04_15_MP4_3_fps_cropped.csv',
                    './dataset/parsed_dataset/01_cropped/20210513_EO_Movie_7_fps_cropped.csv',
                    './dataset/parsed_dataset/01_cropped/20210518_01_7_fps_cropped.csv',
                    './dataset/parsed_dataset/01_cropped/20210518_01_30_fps_cropped.csv',
                    './dataset/parsed_dataset/01_cropped/20210518_02_7_fps_cropped_train.csv',
                    './dataset/parsed_dataset/01_cropped/20210518_02_7_fps_cropped_val.csv',
                    './dataset/parsed_dataset/01_cropped/20210525_EO_Movie_7_fps_cropped_train.csv',
                    './dataset/parsed_dataset/01_cropped/20210525_EO_Movie_7_fps_cropped_val.csv',
                    './dataset/parsed_dataset/01_cropped/20210526_EO_Movie_7_fps_cropped_train.csv',
                    './dataset/parsed_dataset/01_cropped/20210526_EO_Movie_7_fps_cropped_val.csv',
                    './dataset/parsed_dataset/01_cropped/20210601_EO_Movie_7_fps_cropped_train.csv',
                    './dataset/parsed_dataset/01_cropped/20210601_EO_Movie_7_fps_cropped_val.csv',
                    './dataset/parsed_dataset/01_cropped/20210603_EO_Movie_7_fps_cropped_train.csv',
                    './dataset/parsed_dataset/01_cropped/20210603_EO_Movie_7_fps_cropped_val.csv',
                    './dataset/201105_sortie/orig_video_3_fps_labels_no_neg_cropped.csv',
                    './dataset/2021Apr15/2021_04_15_MP4_3_fps_cropped.csv']

    total_file_num = 0
    total_label_count = 0

    for csv_file in IN_CSV_LIST:
       csv_dict = parse_csv_file(csv_file)
       file_num, label_count = data_dict_statictics(csv_dict)

       print(f'for csvfile {csv_file}, image count = {file_num}, label count = {label_count}')

       total_file_num += file_num
       total_label_count += label_count

    print(f'total images : {total_file_num}, total_label_count = {total_label_count}')
