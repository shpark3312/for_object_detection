import math
import os
import argparse
import random
import time
import datetime
import json
import torch
from PIL import ImageFont, ImageDraw, Image
from tqdm import trange
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from template_matching import TemplateMatching


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class ObjectDetection:
    def __init__(self, model_path):
        torch.cuda.device(0)
        self.cuda = torch.device("cuda")
        self.input_size = 832
        self.model_yolo = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        self.model_yolo = self.model_yolo.cuda()

    def detect(self, x, threshold=0.5):
        self.model_yolo.conf = threshold

        results = self.model_yolo(x, size=self.input_size)

        return results.xyxy


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


def nms(boxes, probs, labels, overlap_thresh):
    if not boxes.size:
        return np.empty((0,), dtype=int)
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(probs)
    while idxs.size:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int"), probs[pick], labels[pick]


def crop(img, patch_width=1280, patch_height=720, x_overlay=80, y_overlay=45):
    x_step = patch_width - x_overlay
    y_step = patch_height - y_overlay
    height, width, _ = img.shape
    patches = []
    start_coords = []

    for x_offset in range(math.ceil(width / x_step)):
        for y_offset in range(math.ceil(height / y_step)):

            xstart = max(0, x_step * x_offset)
            ystart = max(0, y_step * y_offset)
            xend = min(width, xstart + patch_width)
            yend = min(height, ystart + patch_height)
            patches.append(img[ystart:yend, xstart:xend])
            start_coords.append([xstart, ystart])

    return patches, start_coords


def sec2time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    pattern = r"%02d:%02d:%02d"

    return pattern % (h, m, s)


def main(args):

    CATEGORY_PATH = args.cat_path
    IN_VIDEOS_PATH = args.video_dir
    OUT_VIDEOS_PATH =args.out_video_dir
    MODEL_PATH = args.model_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(OUT_VIDEOS_PATH, exist_ok=True)

    CROP_WIDTH = 832
    CROP_HEIGHT = 832
    CROP_X_OVERLAP = 80
    CROP_Y_OVERLAP = 168
    BBOX_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.5


    CATEGORY_TABLE = parse_categories_file(CATEGORY_PATH)
    REVERSE_CATEGORY_TABLE = {v: k for k, v in CATEGORY_TABLE.items()}

    COLORS = [(matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int) for x in np.linspace(0, 1, len(CATEGORY_TABLE), endpoint=False)]
    COLORS = [tuple([cx.item() for cx in c]) for c in COLORS]

    model = ObjectDetection(MODEL_PATH)

    log_txt = open(os.path.join(OUT_VIDEOS_PATH, "Processing_log.txt"), "w")

    data = {}
    data["BM"] = 3
    data["Drone"] = "DJI"
    data["GPS_latitude"] = None
    data["GPS_longitude"] = None
    data["GPS_altitude"] = None

    for root, _, files in os.walk(IN_VIDEOS_PATH):
        for filename in files:
            if os.path.splitext(filename)[-1].lower() not in [".mp4", ".mov"]:
                continue
            in_filepath = os.path.join(root, filename)
            out_filepath = os.path.join(OUT_VIDEOS_PATH, filename)
            print(f"{in_filepath} -> {out_filepath}")

            in_vid = cv2.VideoCapture(in_filepath)
            fps = in_vid.get(cv2.CAP_PROP_FPS)
            orig_width = in_vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            orig_height = in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

            timestamp = os.path.getmtime(in_filepath)
            datetimeobj = datetime.datetime.fromtimestamp(int(timestamp))

            data["Image_width"] = orig_width
            data["Image_height"] = orig_height
            data["DateTime"] = str(datetimeobj)
            data["Name"] = filename

            out_vid = cv2.VideoWriter(out_filepath,cv2.VideoWriter_fourcc(*"mp4v"),fps,(int(orig_width), int(orig_height)),)
            count = 1
            current_time = time.time()

            max_frame_count = int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in trange(max_frame_count):
                ret, frame = in_vid.read()
                # cv2.imshow('',frame)
                # cv2.waitKey(0)
                if not ret:
                    break
                draw = frame.copy()
                frame = frame[:, :, ::-1]

                if i % round(fps) == 0:
                    trackers = []
                    cropped_patches, start_coords = crop(
                        frame, CROP_WIDTH, CROP_HEIGHT, CROP_X_OVERLAP, CROP_Y_OVERLAP
                    )

                    result_from_model = model.detect(cropped_patches, BBOX_THRESHOLD)
                    # print(result_from_model)
                    boxes = []
                    scores = []
                    labels = []

                    for res, coords in zip(result_from_model, start_coords):
                        res = res.cpu().numpy()
                        if res is not None:
                            res[:,0] += coords[0]
                            res[:,1] += coords[1]
                            res[:,2] += coords[0]
                            res[:,3] += coords[1]
                            for box in res:
                                boxes.append(box[:4])
                                scores.append(box[4])
                                labels.append(box[5])


                    if len(boxes) != 0:
                        boxes = np.asarray(boxes)
                        scores = np.asarray(scores)
                        labels = np.asarray(labels)

                        boxes, scores, labels = nms(boxes, scores, labels, overlap_thresh=NMS_THRESHOLD)

                        data["Frame"] = i
                        data["comment"] = []
                        data["comment"].append("Person")
                        data["num_detected"] = len(boxes)
                        data["objects_info"] = []

                        for box, score, label in zip(boxes, scores, labels):
                            b = box.astype(int)
                            l = int(label)
                            trackers.append((TemplateMatching(frame, b), l, score))
                            color = COLORS[l]
                            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, 3, cv2.LINE_AA)
                            caption = "{} {:.3f}".format(REVERSE_CATEGORY_TABLE[l], score)
                            font = ImageFont.truetype("./NanumGothicBold.ttf", 20)
                            draw_pil = Image.fromarray(draw)
                            draw_pil_d = ImageDraw.Draw(draw_pil)
                            width, height = draw_pil_d.textsize(caption, font)
                            draw_pil_d.rectangle(((b[0], b[1] - height), b[0] + width, b[1]), fill=color)
                            draw_pil_d.text((b[0], b[1] - height),caption,font=font,fill=(0, 0, 0),)
                            draw = np.array(draw_pil)
                            data["objects_info"].append({
                                    "class": REVERSE_CATEGORY_TABLE[l],
                                    "probability": round(score, 2),
                                    "coordinate": [b[0], b[1], b[2], b[3]],
                                                        })

                else:
                    delete = []
                    for i in trackers:
                        tracker, l, score = i
                        next_rect = tracker.track(frame)
                        if next_rect is None:
                            delete.append(i)
                            continue
                        color = COLORS[l]
                        cv2.rectangle(
                            draw,
                            (next_rect[0], next_rect[1]),
                            (next_rect[2], next_rect[3]),
                            color,
                            3,
                            cv2.LINE_AA,
                        )
                        caption = "{} {:.3f}".format(REVERSE_CATEGORY_TABLE[l], score)
                        font = ImageFont.truetype("./NanumGothicBold.ttf", 20)
                        draw_pil = Image.fromarray(draw)
                        draw_pil_d = ImageDraw.Draw(draw_pil)
                        width, height = draw_pil_d.textsize(caption, font)
                        draw_pil_d.rectangle(((next_rect[0], next_rect[1] - height),next_rect[0] + width,next_rect[1]),fill=color,)
                        draw_pil_d.text((next_rect[0], next_rect[1] - height),caption,font=font,fill=(0, 0, 0),)
                        draw = np.array(draw_pil)

                    for i in delete:
                        trackers.remove(i)

                out_vid.write(draw)

            print(f"{filename} 처리 시간 : {time.time() - current_time}")
            print(f"{filename} test 완료")
            log_txt.write(f"Filename : {filename}\n")
            log_txt.write(f"resolutiuon : {int(orig_width)}x{int(orig_height)}\n")
            log_txt.write(f"video length : {sec2time(round(max_frame_count/60))}\n")
            log_txt.write(f"processing time : {sec2time(round(time.time() - current_time))}\n")
            log_txt.write(f"frames per sec : {max_frame_count / round(time.time() - current_time):.2f}\n\n")

            in_vid.release()
            out_vid.release()

    log_txt.close()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", help="Set in-video dir")
    parser.add_argument("--out_video_dir", help="Set in-image dir")
    parser.add_argument("--model_path", help="Set in-model path")
    parser.add_argument("--cat_path", default = './dataset/classes.csv', help="Set category csv path path")
    parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")
    args = parser.parse_args()

    main(args)
