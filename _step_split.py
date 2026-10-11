# -*- coding: utf-8 -*-
"""把 step_flow.jpg 按行切开，便于逐行查看。"""
import os
import cv2

SRC = r'vis_out2/step_flow.jpg'
OUT = r'vis_out2'

img = cv2.imread(SRC)
h, w = img.shape[:2]
print('step_flow.jpg:', w, 'x', h)

# 每行高度：(1200, h)  h = 1200? no
print('rows split into 5')

row_h = h // 5
for i in range(5):
    y0 = i * row_h
    y1 = h if i == 4 else (i + 1) * row_h
    crop = img[y0:y1]
    # 放大 1.6 倍便于阅读
    crop = cv2.resize(crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    p = os.path.join(OUT, f'step_row{i}.jpg')
    cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(' ->', p, crop.shape[1], 'x', crop.shape[0])

# 热力图单张放大
hm = cv2.imread(r'vis_out2/step_heatmap.jpg')
if hm is not None:
    big = cv2.resize(hm, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(r'vis_out2/step_heatmap_big.jpg', big, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(' -> vis_out2/step_heatmap_big.jpg', big.shape[1], 'x', big.shape[0])
