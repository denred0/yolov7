import shutil
import time
import torch
import cv2
import contextlib
import numpy as np

from typing import List
from pathlib import Path


def get_all_files_in_folder(folder: str, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(Path(folder).rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def recreate_folders(root_dir: Path, folders_list: List) -> None:
    for directory in folders_list:
        output_dir = root_dir.joinpath(directory)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def recreate_folder(root_dir: str) -> None:
    output_dir = Path(root_dir)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def plot_one_box(im, box, label=None, color=(255, 255, 0), line_thickness=1, write_label=True):
    c1 = (box[0], box[1])
    c2 = (box[2], box[3])

    tl = line_thickness or round(0.001 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    im = cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        im = cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        im = cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im


def xyxy2xywhn_single(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = ((x[0] + x[2]) / 2) / w  # x center
    y[1] = ((x[1] + x[3]) / 2) / h  # y center
    y[2] = (x[2] - x[0]) / w  # width
    y[3] = (x[3] - x[1]) / h  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y

    y = [int(a) for a in y]
    return y
