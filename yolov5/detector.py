from .utils.torch_utils import select_device
from .utils.plots import plot_one_box
from .utils.boxes import non_max_suppression, scale_coords
from .data.augmentations import letterbox
from .models.experimental import attempt_load
import random
import os
import cv2
import numpy as np
import torch

random.seed(0)


class Yolov5:
    def __init__(self, weight_path, device, img_hw=(384, 640)):
        self.weights = weight_path
        self.device = select_device(device)
        self.half = True
        # path aware
        with torch.no_grad():
            self.model = attempt_load(self.weights, map_location=self.device)

        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]
        self.show = False
        self.img_hw = img_hw
        self.pause = False

    def preprocess(self, image, auto=True):  # (h, w)
        if type(image) == str and os.path.isfile(image):
            img0 = cv2.imread(image)
        else:
            img0 = image
        img, _, _ = letterbox(img0, new_shape=self.img_hw, auto=auto)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0

    @torch.no_grad()
    def dynamic_detect(self, image, img0s, classes=None, conf_threshold=0.6, iou_threshold=0.4):
        output = {}
        if classes is not None:
            for c in classes:
                output[self.names[int(c)]] = 0
        else:
            for n in self.names:
                output[n] = 0
        img = torch.from_numpy(image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        torch.cuda.synchronize()
        pred = self.model(img)[0] 
        pred = non_max_suppression(
            pred, conf_threshold, iou_threshold, classes=classes, agnostic=False)

        torch.cuda.synchronize()
        for i, det in enumerate(pred):  # detections per image
            if det is None or len(det) == 0:
                continue
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0s[i].shape).round()

            for _, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                output[self.names[int(cls)]] += 1
                # label = '%s %.2f' % (self.names[int(cls)], conf)
                # label = '%s' % (self.names[int(cls)])
                label = None
                color = [0, 0, 255] if conf < 0.6 else self.colors[int(cls)]
                if self.show:
                    plot_one_box(xyxy, img0s[i], label=label,
                                 color=color, 
                                 line_thickness=2)

        if self.show:
            for i in range(len(img0s)):
                cv2.namedWindow(f'p{i}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'p{i}', img0s[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(' ') else False
            if key == ord('q') or key == ord('e') or key == 27:
                exit()
        return pred, output


if __name__ == '__main__':
    detector = Yolov5(weight_path='', device='0', img_hw=(640, 640))

    detector.show = True
    detector.pause = True
    # for video
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     img, img_raw = detector.preprocess(frame, auto=True)
    #     preds, _ = detector.dynamic_detect(img, [img_raw])