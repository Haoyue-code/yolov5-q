"""
Read model's output from numpy(.npy) and do some postprocessing.
input:
    output1: (1, 3, 80, 80, 85)
    output2: (1, 3, 40, 40, 85)
    output3: (1, 3, 20, 20, 85)
"""

import numpy as np
import cv2
from yolov5.utils.boxes import non_max_suppression_numpy, scale_coords
from yolov5.utils.plots import Visualizer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TestPost:
    def __init__(self, preds=[]) -> None:
        self.stride = [8, 16, 32]
        self.preds = [np.load(p) for p in preds]
        self.nl = len(self.preds)
        self.na = self.preds[0].shape[1]
        self.no = self.preds[0].shape[-1]

        self.grid = [np.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [np.zeros(1)] * self.nl  # init anchor grid
        self.anchors = (
            np.array(
                [
                    [10, 13, 16, 30, 33, 23],
                    [30, 61, 62, 45, 59, 119],
                    [116, 90, 156, 198, 373, 326],
                ]
            )
            .astype(np.float32)
            .reshape(self.nl, -1, 2)
        )
        self.vis = Visualizer(names=list(range(self.no - 5)))

    def __call__(self):
        z = []
        for i in range(self.nl):
            _, _, ny, nx, _ = self.preds[i].shape  # (bs, 3, 20, 20, 85)
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            print(self.grid[i].shape, self.anchor_grid[i].shape)
            y = sigmoid(self.preds[i])
            xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            y = np.concatenate((xy, wh, y[..., 4:]), -1)
            z.append(y.reshape(-1, self.na * ny * nx, self.no))
        output = np.concatenate(z, 1)
        return output

    def _make_grid(self, nx=20, ny=20, i=0):
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
        grid = np.stack((xv.T, yv.T), 2)[None, None, ...].repeat(3, axis=1).astype(np.float32)
        anchor_grid = (
            (self.anchors[i].copy())
            .reshape((1, self.na, 1, 1, 2))
            .repeat(ny, 2)
            .repeat(nx, 3)
            .astype(np.float32)
        )
        return grid, anchor_grid


if __name__ == "__main__":
    preds = [
        "./outputs_int8/147.npy",
        "./outputs_int8/148.npy",
        "./outputs_int8/149.npy",
    ]
    # preds = [
    #     "./outputs_fp32/147.npy",
    #     "./outputs_fp32/148.npy",
    #     "./outputs_fp32/149.npy",
    # ]
    test = TestPost(preds=preds)
    output = test()
    print(output.shape)
    output = non_max_suppression_numpy(output, conf_thres=0.2)
    print(output)

    name = 'smoke_phone.jpg'
    img = cv2.imread(f'./test_imgs/{name}')

    for i, det in enumerate(output):  # detections per image
        if det is None or len(det) == 0:
            continue
        det[:, :4] = scale_coords((416, 768), det[:, :4], img.shape[:2]).round()

    print(output)
    # img = test.vis(img, output, vis_confs=0.0)
    # # cv2.imwrite(f'int8/{name}', img)
    # cv2.imshow('p', img)
    # cv2.waitKey(0)
