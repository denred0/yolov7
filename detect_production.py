from pathlib import Path

import cv2
import torch
import numpy as np
from numpy import random
from collections import defaultdict

from tqdm import tqdm
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel

from helpers import recreate_folder, get_all_files_in_folder, Profile
from detect_production.map import mean_average_precision


def get_model(device, weights, model_imgsz, trace, half):
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(model_imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model, names, colors


def detect(source: str,
           images_ext: str,
           weights: str,
           save_dir: str,
           annot_save_dir: str,
           model_imgsz: int,
           conf_thres: float,
           iou_thres: float,
           trace: bool,
           map_iou: float,
           map_calc: bool,
           draw_detections: bool,
           verbose: bool,
           half: bool,
           draw_gt: bool,
           save_img=True,
           save_txt=True,
           augment=False,
           save_conf=True,
           draw_label=False,
           print_speed=False) -> None:
    save_dir = Path(save_dir)
    annot_save_dir = Path(annot_save_dir)

    images = get_all_files_in_folder(source, [f'*.{images_ext}'])

    map_images = []
    map_classes_total = defaultdict(list)
    precision_images = []
    recall_images = []

    dt = (Profile(), Profile(), Profile())

    # Initialize
    device = select_device('')
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    model, names, colors = get_model(device, weights, model_imgsz, trace, half)

    for im in tqdm(images):
        img0 = cv2.imread(str(im))
        h, w = img0.shape[:2]
        assert img0 is not None, f'Image Not Found {im}'

        detections_result = []

        # Preprocessing
        with dt[0]:
            img = preprocess_image(img0, device, model_imgsz, half)

        # Inference
        with torch.no_grad():
            with dt[1]:
                pred = model(img, augment=augment)[0]

        # Apply NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # Print time (inference + NMS)
        if print_speed:
            print(
                f'Done. ({dt[0].dt * 1E3:.1f}ms) Preprocess, ({dt[1].dt * 1E3:.1f}ms) Inference, ({dt[2].dt * 1E3:.1f}ms) NMS')

        save_path = str(save_dir / im.name)  # img.jpg
        txt_path = str(annot_save_dir / f'{im.stem}.txt')
        open(txt_path, 'x').close()
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    if save_txt:  # Write to file
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    detections_result.append(
                        [
                            int(cls),
                            round(conf.cpu().numpy().item(), 2),
                            xywh[0],
                            xywh[1],
                            xywh[2],
                            xywh[3]
                        ])

                    if draw_detections:
                        if draw_label:
                            label = f'{names[int(cls)]} {conf:.2f}'
                        else:
                            label = ''
                        # Add bbox to image
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

        if map_calc:
            with open(Path(source).joinpath(im.stem + ".txt")) as file:
                detections_gt = file.readlines()
                detections_gt = [d.replace("\n", "") for d in detections_gt]
                detections_gt = [d.split() for d in detections_gt]
                detections_gt = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])] for d in detections_gt]

            if detections_result == detections_gt == []:
                map_images.append(1)
                precision_images.append(1)
                recall_images.append(1)
            elif detections_gt == [] and detections_result != []:
                map_images.append(0)
                precision_images.append(0)
                recall_images.append(0)
            else:
                map_image, precision_image, recall_image, map_classes = mean_average_precision(
                    pred_boxes=detections_result,
                    true_boxes=detections_gt,
                    num_classes=len(names),
                    iou_threshold=map_iou)
                map_images.append(map_image)
                precision_images.append(precision_image)
                recall_images.append(recall_image)

                for cl, ap in map_classes.items():
                    map_classes_total[cl].append(ap)

        if draw_gt:
            for det_gt in detections_gt:
                x_top = int((det_gt[1] - det_gt[3] / 2) * w)
                y_top = int((det_gt[2] - det_gt[4] / 2) * h)
                x_bottom = int((x_top + det_gt[3] * w))
                y_bottom = int((y_top + det_gt[4] * h))

                class_gt = names[det_gt[0]]

                plot_one_box(
                    [int(x_top), int(y_top), int(x_bottom), int(y_bottom)],
                    img0,
                    color=(0, 255, 0),
                    label=str(class_gt),
                    line_thickness=1
                )

        # Save results (image with detections)
        if save_img:
            cv2.imwrite(save_path, img0)

    if verbose:
        print(f"Images count: {len(images)}")
        print(f"mAP: {round(np.mean(map_images), 4)}")
        # precision - не находим лишнее (уменьшаем FP)
        print(f"Precision: {round(np.mean(precision_images), 4)}")
        # recall - находим все объекты (уменьшаем FN)
        print(f"Recall: {round(np.mean(recall_images), 4)}")

        print()
        for key, value in dict(sorted(map_classes_total.items())).items():
            print(f"{names[key]}: {round(sum(value) / len(value), 4)}")

        print()
        print(f"th: {conf_thres}")
        print(f"iou_thres: {iou_thres}")
        print(f"mAP IoU: {map_iou}")
        t = tuple(x.t / len(images) * 1E3 for x in dt)  # speeds per image
        print(f"FPS: {round(len(images) / (dt[0].t + dt[1].t + dt[2].t), 2)}")
        print(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, model_imgsz, model_imgsz)}' % t)


def preprocess_image(img0, device, model_imgsz, half):
    img = letterbox(img0, model_imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == '__main__':
    project = 'notes'

    weights = f'data/detect_production/{project}/input/cfg/best.pt'

    source = f'data/detect_production/{project}/input/gt_images_txts'
    images_ext = 'jpg'

    save_dir = f'data/detect_production/{project}/output/images_vis'
    recreate_folder(save_dir)

    annot_save_dir = f"data/detect_production/{project}/output/annot_pred"
    recreate_folder(annot_save_dir)

    map_iou = 0.8
    map_calc = True

    conf_thres = 0.4
    iou_thres = 0.45

    model_imgsz = 640
    trace = True
    verbose = True
    half = True
    draw_gt = False
    draw_detections = True

    detect(source,
           images_ext,
           weights,
           save_dir,
           annot_save_dir,
           model_imgsz,
           conf_thres,
           iou_thres,
           trace,
           map_iou,
           map_calc,
           draw_detections,
           verbose,
           half,
           draw_gt)
