"""用 config.py 真正构建 dataloader，跑几个 batch 验证全链路可用。"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner


def main():
    init_default_scope('mmpose')

    cfg = Config.fromfile('config.py')
    print('config 解析成功')
    print('  input_size   =', cfg.input_size)
    print('  heatmap_size =', cfg.heatmap_size)
    print('  codec        =', cfg.codec['type'], 'sigma=', cfg.codec['sigma'])
    print('  train pipeline 长度 =', len(cfg.train_pipeline))
    print('  val   pipeline 长度 =', len(cfg.val_pipeline))
    print('  test  pipeline 长度 =', len(cfg.test_pipeline))
    print()

    for name in ('train', 'val', 'test'):
        dl_cfg = cfg[f'{name}_dataloader']
        ds_cfg = dl_cfg['dataset']
        print(f'--- {name}_dataloader ---')
        print(f'  batch_size={dl_cfg["batch_size"]}  '
              f'num_workers={dl_cfg["num_workers"]}  '
              f'ann_file={ds_cfg["ann_file"]!r}')

        loader = Runner.build_dataloader(dl_cfg, seed=0,
                                         diff_rank_seed=False)
        ds = loader.dataset
        print(f'  数据集实例数 = {len(ds)}')
        print(f'  metainfo.num_keypoints = {ds.metainfo.get("num_keypoints")}')
        print(f'  metainfo.sigmas = {ds.metainfo.get("sigmas")}')

        try:
            batch = next(iter(loader))
        except KeyError as e:
            if name == 'test' and 'bbox' in str(e):
                print()
                print('  [预期] 测试集取 batch 失败: KeyError("bbox")')
                print('    这是【预期行为】: 测试集图片没有标注，也就没有 bbox；')
                print('    而 top-down 流水线第一步 GetBBoxCenterScale 需要 bbox。')
                print('    => 推理前必须先由【目标检测器】给出每个 DM 码的框，')
                print('       把 bbox 塞进 data_info 再走本流水线。')
                print()
                continue
            raise

        inputs = batch['inputs']
        dsamples = batch['data_samples']
        if isinstance(inputs, (list, tuple)):
            ishape = f'list[{len(inputs)}] x {tuple(inputs[0].shape)}'
        else:
            ishape = tuple(inputs.shape)
        print(f'  一个 batch: inputs={ishape}  '
              f'data_samples={len(dsamples)} 个')
        s0 = dsamples[0]
        print(f'    第0个: img_id={s0.metainfo.get("img_id")}  '
              f'input_size={s0.metainfo.get("input_size")}')
        if getattr(s0, 'gt_fields', None) is not None:
            print(f'    gt_fields.heatmaps = '
                  f'{tuple(s0.gt_fields.heatmaps.shape)}')
            print(f'    keypoint_weights   = '
                  f'{s0.gt_instance_labels.keypoint_weights.tolist()}')
        else:
            print('    (测试集无 GT，只有 img + metainfo)')
        print()

    print('train / val 通过。test 的 bbox 需由检测器提供（见上）。')


if __name__ == '__main__':
    main()
