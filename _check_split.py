from mmengine.registry import init_default_scope
init_default_scope('mmpose')
from mmpose.datasets.datasets.DM import DMKPD_dataset

META = dict(from_file='open-mmlab/mmpose/configs/_base_/datasets/DMKPD_meta.py')
for split in ('train', 'val'):
    ds = DMKPD_dataset(
        data_root='DMKPD_dataset',
        ann_file=f'splits/{split}.txt',
        data_prefix=dict(img='train/images'),
        metainfo=META,
        serialize_data=False)
    ids = [ds.get_data_info(i)['img_id'] for i in range(3)]
    print(f'{split}.txt -> {len(ds)} 实例;  前几个 img_id = {ids}')

# 交叉检查：两个 split 的 img_id 集合不能有交集
ds_tr = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/train.txt',
    data_prefix=dict(img='train/images'), metainfo=META, serialize_data=False)
ds_va = DMKPD_dataset(
    data_root='DMKPD_dataset', ann_file='splits/val.txt',
    data_prefix=dict(img='train/images'), metainfo=META, serialize_data=False)

tr_imgs = {ds_tr.get_data_info(i)['img_id'] for i in range(len(ds_tr))}
va_imgs = {ds_va.get_data_info(i)['img_id'] for i in range(len(ds_va))}
print()
print(f'train 图数 = {len(tr_imgs)}')
print(f'val   图数 = {len(va_imgs)}')
print(f'交集图数   = {len(tr_imgs & va_imgs)}   (必须为 0)')
print(f'合计图数   = {len(tr_imgs) + len(va_imgs)}')
