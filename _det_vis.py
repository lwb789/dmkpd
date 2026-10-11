# -*- coding: utf-8 -*-
"""看一眼测试集与训练集 1702 档的实拍图，确认检测难度与目标数量。"""
import os
import glob

import cv2
import numpy as np

CELL = 560


def fit(im, cell=CELL):
    h, w = im.shape[:2]
    sc = min(cell / w, cell / h)
    rs = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    out = np.full((cell, cell, 3), 20, np.uint8)
    oy, ox = (cell - rs.shape[0]) // 2, (cell - rs.shape[1]) // 2
    out[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs
    return out


def bar(t, cell=CELL):
    b = np.full((40, cell, 3), 45, np.uint8)
    cv2.putText(b, t, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (255, 255, 255), 2)
    return b


# 测试集 6 张
tests = sorted(glob.glob('DMKPD_dataset/test/*.jpg'))[:6]
row = [np.vstack([bar(os.path.basename(p)), fit(cv2.imread(p))]) for p in tests]
cv2.imwrite('vis_out2/det_test_samples.jpg', np.hstack(row[:3]))
cv2.imwrite('vis_out2/det_test_samples2.jpg', np.hstack(row[3:6]))
print('test imgs -> vis_out2/det_test_samples.jpg / _2.jpg')

# 训练集 1702 档 6 张（带 GT 框）
import json
d = json.load(open('det_dataset/annotations/instances_train.json',
                   encoding='utf-8'))
big = [im for im in d['images'] if im['width'] > 1400]
by_img = {}
for a in d['annotations']:
    by_img.setdefault(a['image_id'], []).append(a)
sel = [im for im in big if len(by_img.get(im['id'], [])) >= 3][:6]
print('选中 %d 张多目标图' % len(sel))
row = []
for im in sel:
    p = os.path.normpath(os.path.join('det_dataset',
                                      im['file_name']))
    img = cv2.imread(p)
    for a in by_img[im['id']]:
        x, y, w, h = a['bbox']
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 255, 0), 6)
    row.append(np.vstack([bar(os.path.basename(p)), fit(img)]))
cv2.imwrite('vis_out2/det_train_samples.jpg', np.hstack(row[:3]))
cv2.imwrite('vis_out2/det_train_samples2.jpg', np.hstack(row[3:6]))
print('train imgs -> vis_out2/det_train_samples.jpg / _2.jpg')

# 512 档看一下
sm = [im for im in d['images'] if im['width'] <= 600][:3]
row = []
for im in sm:
    p = os.path.normpath(os.path.join('det_dataset', im['file_name']))
    img = cv2.imread(p)
    for a in by_img[im['id']]:
        x, y, w, h = a['bbox']
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 255, 0), 3)
    row.append(np.vstack([bar(os.path.basename(p)), fit(img)]))
cv2.imwrite('vis_out2/det_train_512.jpg', np.hstack(row))
print('512 imgs -> vis_out2/det_train_512.jpg')
