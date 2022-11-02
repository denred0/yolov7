import cv2
import torch
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from helpers import recreate_folder, get_all_files_in_folder, Profile
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box

from helpers import xywhn2xyxy, xyxy2xywhn_single


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


def get_image_parts(img, desired_size):
    h, w = img.shape[:2]

    parts_w = w // desired_size[0] + 1
    extra_w = (parts_w * desired_size[0] - w)
    one_part_inside_w = (w - extra_w) // parts_w
    one_part_extra_w = 0
    if parts_w > 1:
        one_part_extra_w = (w - one_part_inside_w * parts_w) // (parts_w - 1)

    parts_h = h // desired_size[1] + 1
    extra_h = (parts_h * desired_size[1] - h)
    one_part_inside_h = (h - extra_h) // parts_h
    one_part_extra_h = 0
    if parts_h > 1:
        one_part_extra_h = (h - one_part_inside_h * parts_h) // (parts_h - 1)

    min_x, min_y = 0, 0
    max_x = min(desired_size[0], w)
    max_y = min(desired_size[1], h)

    parts_images = []
    sizes = []
    for i in (range(parts_w)):
        for j in (range(parts_h)):
            parts_images.append([min_x, min_y, max_x, max_y, i, j])
            sizes.append((max_x - min_x) * (max_y - min_y))

            min_y = max_y - one_part_extra_h
            max_y = min_y + min(desired_size[1], h)

        min_x = max_x - one_part_extra_w
        max_x = min_x + min(desired_size[0], w)

        min_y = 0
        max_y = min(desired_size[1], h)

    return parts_images, one_part_extra_w, one_part_extra_h


def get_image_size(cut_size_x, cut_size_y, max_ind_x, max_ind_y, extra_part_x, extra_part_y, h, w):
    min_x = 0
    min_y = 0
    max_x = w if max_ind_x == 0 else cut_size_x
    max_y = h if max_ind_y == 0 else cut_size_y
    for i in range(max_ind_x):
        for j in range(max_ind_y):
            min_y = max_y - extra_part_y
            add_y = h if max_ind_y == 0 else cut_size_y
            max_y = min_y + add_y

        min_x = max_x - extra_part_x
        add_x = w if max_ind_x == 0 else cut_size_x
        max_x = min_x + add_x

        min_y = 0
        if i != max_ind_x - 1:
            max_y = h if max_ind_y == 0 else cut_size_y

    return max_x, max_y


def get_transformed_annotation(an, cut_size_x, cut_size_y, min_x, min_y, w, h):
    an = an.split()
    an = [float(a) for a in an]
    an_xyxy = xywhn2xyxy(an[1:], w=cut_size_x, h=cut_size_y)

    an_xyxy_w = an_xyxy[2] - an_xyxy[0]
    an_xyxy_h = an_xyxy[3] - an_xyxy[1]

    x1_new = an_xyxy[0] + min_x
    y1_new = an_xyxy[1] + min_y
    x2_new = x1_new + an_xyxy_w
    y2_new = y1_new + an_xyxy_h

    an_new_xyxyn = xyxy2xywhn_single(
        np.asarray([x1_new, y1_new, x2_new, y2_new], dtype=np.float64),
        w=w,
        h=h
    )

    return [int(an[0]), an_new_xyxyn[0], an_new_xyxyn[1], an_new_xyxyn[2], an_new_xyxyn[3]]


def merge_part_images(input_dir, images_ext, output_dir):
    images = get_all_files_in_folder(input_dir, [f'*.{images_ext}'])
    annotations = get_all_files_in_folder(input_dir, ['*.txt'])

    assert len(images) == len(annotations), 'len(images) != len(annotations)'

    unique_images = list(set([a.stem.split('_')[:-6][0] for a in images]))

    for unique_im in tqdm(unique_images):

        parts = {}
        max_ind_x = 0
        max_ind_y = 0
        cut_size_x = 0
        cut_size_y = 0
        extra_part_x = 0
        extra_part_y = 0

        image_annotations = []

        for im in images:
            img_name = im.stem.split('_')[:-6][0]

            if img_name == unique_im:
                x_ind = int(im.stem.split('_')[-2])
                max_ind_x = max(max_ind_x, x_ind)
                y_ind = int(im.stem.split('_')[-1])
                max_ind_y = max(max_ind_y, y_ind)

                cut_size_x = int(im.stem.split('_')[-6])
                cut_size_y = int(im.stem.split('_')[-5])
                extra_part_x = int(im.stem.split('_')[-4])
                extra_part_y = int(im.stem.split('_')[-3])

                parts[f'{str(x_ind)}_{str(y_ind)}'] = im

        min_x = 0
        min_y = 0

        one_part = cv2.imread(str(parts['0_0']))
        max_x = one_part.shape[1] if max_ind_x == 0 else cut_size_x
        max_y = one_part.shape[0] if max_ind_y == 0 else cut_size_y

        w, h = get_image_size(
            cut_size_x,
            cut_size_y,
            max_ind_x,
            max_ind_y,
            extra_part_x,
            extra_part_y,
            one_part.shape[0],
            one_part.shape[1]
        )
        blank_image = np.zeros((h, w, 3), np.uint8)
        for i in range(max_ind_x + 1):
            for j in range(max_ind_y + 1):
                path_img_part = parts[f'{str(i)}_{str(j)}']

                img = cv2.imread(str(path_img_part))
                blank_image[min_y:max_y, min_x:max_x] = img

                with open(path_img_part.parent.joinpath(f'{path_img_part.stem}.txt')) as file:
                    annotations = file.readlines()
                    annotations = [line.rstrip() for line in annotations]

                for an in annotations:
                    transformed_annotation = get_transformed_annotation(
                        an,
                        one_part.shape[1] if max_ind_x == 0 else cut_size_x,
                        one_part.shape[0] if max_ind_y == 0 else cut_size_y,
                        min_x,
                        min_y,
                        w,
                        h
                    )

                    image_annotations.append(transformed_annotation)

                min_y = max_y - extra_part_y
                max_y = min_y + cut_size_y

            min_x = max_x - extra_part_x
            max_x = min_x + cut_size_x

            min_y = 0
            max_y = cut_size_y

        cv2.imwrite(str(Path(output_dir).joinpath(f'{unique_im}.{images_ext}')), blank_image)

        with open(Path(output_dir).joinpath(f'{unique_im}.txt'), 'w') as f:
            for annot in image_annotations:
                f.write(f"{annot[0]} {annot[1]} {annot[2]} {annot[3]} {annot[4]}\n")


def detect(
        input_dir: str,
        images_ext: str,
        desired_size: tuple,
        save_dir: str,
        annot_save_dir: str,
        model_imgsz,
        trace,
        half,
        conf_thres,
        iou_thres,
        draw_detections,
        save_img,
        save_txt=True,
        save_conf=True
):
    dt = (Profile(), Profile(), Profile())

    # Initialize
    device = select_device('')
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    model, names, colors = get_model(device, weights, model_imgsz, trace, half)

    images = get_all_files_in_folder(input_dir, [f'*.{images_ext}'])

    for im in tqdm(images):
        img0 = cv2.imread(str(im))
        img_orig = img0.copy()
        assert img0 is not None, f'Image Not Found {im}'

        parts_images, one_part_extra_w, one_part_extra_h = get_image_parts(img0, desired_size)

        for i, part in enumerate(parts_images):
            image_part = img0[part[1]:part[3], part[0]:part[2]]
            image_part_orig = img_orig[part[1]:part[3], part[0]:part[2]]

            detections_result = []

            # Preprocessing
            with dt[0]:
                img = preprocess_image(image_part, device, model_imgsz, half)

            # Inference
            with torch.no_grad():
                with dt[1]:
                    pred = model(img, augment=False)[0]

            # Apply NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

            # Print time (inference + NMS)
            print(
                f'Done. ({dt[0].dt * 1E3:.1f}ms) Preprocess, ({dt[1].dt * 1E3:.1f}ms) Inference, ({dt[2].dt * 1E3:.1f}ms) NMS')

            save_path_vis = Path(save_dir).joinpath(
                f'{im.stem}_{desired_size[0]}_{desired_size[1]}_{one_part_extra_w}_{one_part_extra_h}_{part[4]}_{part[5]}.{images_ext}')
            save_path = Path(annot_save_dir).joinpath(
                f'{im.stem}_{desired_size[0]}_{desired_size[1]}_{one_part_extra_w}_{one_part_extra_h}_{part[4]}_{part[5]}.{images_ext}')
            cv2.imwrite(str(save_path), image_part_orig)

            txt_path = Path(annot_save_dir).joinpath(
                f'{im.stem}_{desired_size[0]}_{desired_size[1]}_{one_part_extra_w}_{one_part_extra_h}_{part[4]}_{part[5]}.txt')
            open(txt_path, 'x').close()
            gn = torch.tensor(image_part.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_part.shape).round()

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

                        if draw_detections:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # if int(cls.item()) == 0:
                            #     xc = int(torch.tensor(xyxy).view(1, 4).detach().numpy()[0][0]) + int(
                            #         (torch.tensor(xyxy).view(1, 4).detach().numpy()[0][2] - \
                            #          torch.tensor(xyxy).view(1, 4).detach().numpy()[0][0]) * 0.62)
                            #     yc = int(torch.tensor(xyxy).view(1, 4).detach().numpy()[0][1]) + int(
                            #         (torch.tensor(xyxy).view(1, 4).detach().numpy()[0][3] - \
                            #          torch.tensor(xyxy).view(1, 4).detach().numpy()[0][1]) * 0.66)
                            #     image_part = cv2.circle(image_part, (xc, yc), 2, (0, 255, 0), -1)
                            plot_one_box(xyxy, image_part, label=None, color=colors[int(cls)], line_thickness=1)

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(str(save_path_vis), image_part)


def plot_yolo_box(input_images_dir, images_ext, output_dir, classes_path, filter_classes):
    images = get_all_files_in_folder((input_images_dir), [f'*.{images_ext}'])

    for im in tqdm(images):

        img = cv2.imread(str(im))
        h, w = img.shape[:2]

        if classes_path is not None:
            with open(classes_path) as file:
                classes = file.readlines()
                classes = [line.rstrip() for line in classes]

        with open(Path(input_images_dir).joinpath(f'{im.stem}.txt')) as file:
            bboxes = file.readlines()
            bboxes = [line.rstrip() for line in bboxes]

        for box_str in bboxes:
            box = [float(x) for x in box_str.split()]

            label_num = str(int(box[0]))
            if classes_path is not None:
                label = classes[int(box[0]) - 1]

            if filter_classes is not None and label not in filter_classes:
                continue

            xmin = int((box[1] - box[3] / 2) * w)
            ymin = int((box[2] - box[4] / 2) * h)
            xmax = int((box[1] + box[3] / 2) * w)
            ymax = int((box[2] + box[4] / 2) * h)

            plot_one_box([xmin, ymin, xmax, ymax], img, [255, 0, 0], label_num, 1)

        cv2.imwrite(str(Path(output_dir).joinpath(im.name)), img)


if __name__ == '__main__':
    project = 'notes'

    input_dir = f'data/detect_part_images/input'
    images_ext = 'jpg'

    save_dir = f'data/detect_part_images/output/parts/visualization'
    recreate_folder(save_dir)
    annot_save_dir = f'data/detect_part_images/output/parts/annotations'
    recreate_folder(annot_save_dir)

    desired_size = (640, 640)

    weights = 'data/detect_part_images/input/cfg/best.pt'
    model_imgsz = 640
    trace = True
    verbose = True
    half = True
    draw_detections = True
    save_img = True

    conf_thres = 0.4
    iou_thres = 0.45

    detect(
        input_dir,
        images_ext,
        desired_size,
        save_dir,
        annot_save_dir,
        model_imgsz,
        trace,
        half,
        conf_thres,
        iou_thres,
        draw_detections,
        save_img
    )

    # merge
    output_merged_dir = 'data/detect_part_images/output/merged/annotations'
    recreate_folder(output_merged_dir)
    merge_part_images(annot_save_dir, images_ext, output_merged_dir)

    # draw predictions
    input_images_dir = output_merged_dir

    output_vis_dir = 'data/detect_part_images/output/merged/visualization'
    recreate_folder(output_vis_dir)
    classes_path = None
    filter_classes = None
    plot_yolo_box(input_images_dir, images_ext, output_vis_dir, classes_path, filter_classes)
