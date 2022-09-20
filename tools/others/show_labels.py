import os
import random
import numpy as np
import torch.nn.functional as F
from yolov5.utils.general import *
from yolov5.utils.boxes import xywh2xyxy
from yolov5.utils.plots import plot_images_boxes_and_masks
from yolov5.data import LoadImagesAndLabelsAndMasks, create_dataloader
import cv2
from yolov5.utils.general import init_seeds

names = ["balloon"]
seed = 2
init_seeds(seed)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

with open("data/hyps/hyp.scratch.yaml") as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

save = False
save_dir = "/d/projects/yolov5/data/play_phone0115/show_labels"

if save and not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataloader, dataset = create_dataloader(
            # '/d/baidubase/COCO/val_yolo/images/train',
            # 'data/license_plates/images/train/',
            # '/d/projects/research/yolov5/data/coco/train2017.txt',
            # '/d/projects/research/yolov5/data/coco/val2017.txt',
            # "/home/laughing/code/yolov5-6/data/seg/coco_02/images/train",
            "/d/dataset/smoke/smoke_pic/images",
            imgsz=640,
            batch_size=16,
            stride=32,
            augment=False,
            shuffle=True,
            rank=-1,
            hyp=hyp,
            mask_head=False
)
cv2.namedWindow("mosaic", cv2.WINDOW_NORMAL)

for i, (imgs, targets, paths, _, masks) in enumerate(dataloader):
    #     print(targets)
    # print(targets)

    # show masks
    # mxywh = targets[:, 2:].cpu()
    # mxyxy = xywh2xyxy(mxywh) * torch.tensor((640, 640))[[0, 1, 0, 1]]
    # maskssss = masks.permute(1, 2, 0).contiguous()
    # for k in range(maskssss.shape[-1]):
    #     c1 = (int(mxyxy[k][0]), int(mxyxy[k][1]))
    #     c2 = (int(mxyxy[k][2]), int(mxyxy[k][3]))
    #     img = maskssss[:, :, k].cpu().numpy().astype(np.uint8) * 255
    #     cv2.rectangle(img, c1, c2, (255, 255, 255), -1, cv2.LINE_AA)  # filled
    #     cv2.imshow('p', cv2.resize(img, (640, 640)))
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    # exit()

    if getattr(dataset, 'downsample_ratio', 1) != 1 and masks is not None:
        masks = F.interpolate(
            masks[None, :],
            (640, 640),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    result = plot_images_boxes_and_masks(images=imgs, targets=targets, paths=paths, masks=masks)
    cv2.imshow("mosaic", result[:, :, ::-1])
    if cv2.waitKey(0) == ord("q"):  # q to quit
        break
    continue
    # imgs = imgs.numpy().astype(np.uint8).transpose((1, 2, 0))
    # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    # targets = targets.numpy()
    # temp = targets[:, 2:6].copy()
    # targets[:, 2:6] = xywh2xyxy(temp * 960)
    # for idx, cls, *xyxy in targets:
    #     label = '%s' % (names[int(cls)])
    #     plot_one_box(xyxy, imgs, label=label, color=colors[int(cls)], line_thickness=2)
    # # print(targets[targets[:, 0] == 0])
    # if save and len(targets[targets[:, 1] == 0]) > 0:
    #     cv2.imwrite(os.path.join(save_dir, os.path.split(paths)[-1]), imgs)
    # cv2.imshow('p', imgs)
    # cv2.waitKey(0)
