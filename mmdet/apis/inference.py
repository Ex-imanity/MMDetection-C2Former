# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model)
        model.cfg = config

    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)

#### 支持双流输入的推理 ####
def _replace_image_to_paired_default(pipeline):
    """Replace any ImageToTensor in test pipeline with PairedImageDefaultFormatBundle,
    so that both img and img_tir are wrapped into DataContainer (stack=True)."""
    def _replace_in_list(pl):
        for i, t in enumerate(pl):
            if isinstance(t, dict):
                if t.get('type') == 'MultiScaleFlipAug' and 'transforms' in t:
                    t['transforms'] = _replace_in_list(t['transforms'])
                elif t.get('type') == 'ImageToTensor':
                    # Replace with PairedImageDefaultFormatBundle
                    pl[i] = dict(type='PairedImageDefaultFormatBundle')
        return pl

    if isinstance(pipeline, list):
        pipeline = _replace_in_list(pipeline)
    return pipeline


def inference_detector_paired(model, imgs_pair):
    """Inference on paired inputs (RGB + TIR).

    Args:
        model (nn.Module): The loaded detector.
        imgs_pair: 
            - tuple(str|ndarray, str|ndarray): (vis, tir) single pair
            - list[tuple(str|ndarray, str|ndarray)]: batch of pairs

    Returns:
        If input is a single pair, returns the detection result (same结构 as原版).
        If input is a list of pairs, returns a list of detection results.
    """
    # Normalize input to a list of pairs
    if isinstance(imgs_pair, (list, tuple)) and len(imgs_pair) > 0 and isinstance(imgs_pair[0], (str, np.ndarray)):
        # single pair (vis, tir)
        pairs = [imgs_pair]
        is_batch = False
    else:
        # batch of pairs
        pairs = imgs_pair
        is_batch = True

    cfg = model.cfg.copy()
    device = next(model.parameters()).device

    # 判断是否是内存数组输入（仅看第一对即可）
    first_vis, first_tir = pairs[0]
    is_ndarray_input = isinstance(first_vis, np.ndarray) and isinstance(first_tir, np.ndarray)

    # 如果是 ndarray 输入，首个 loader 需要改为 LoadImageFromWebcam，以便管线后续正常工作
    # 注意：LoadImageFromWebcam 只处理 'img'，但我们会把 'img_tir' 直接放入 results，后续 transforms 中的
    # ImageToTensor(keys=['img', 'img_tir'])/PairedImageDefaultFormatBundle 会处理两者。
    if is_ndarray_input:
        if isinstance(cfg.data.test.pipeline[0], dict):
            cfg.data.test.pipeline[0]['type'] = 'LoadImageFromWebcam'

    # 将 pipeline 中的 ImageToTensor 替换为 PairedImageDefaultFormatBundle，
    # 以保证两个模态都打包为 DataContainer（和原版 inference 的 replace_ImageToTensor 作用一致但更适配双模态）
    cfg.data.test.pipeline = _replace_image_to_paired_default(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for vis, tir in pairs:
        if isinstance(vis, np.ndarray) and isinstance(tir, np.ndarray):
            # 直接提供数组：跳过 LoadPairedImageFromFile，使用 LoadImageFromWebcam 加载 'img'
            data = dict(img=vis, img_tir=tir)
        else:
            # 提供文件路径：由 LoadPairedImageFromFile 读取两模态
            data = dict(img_info=dict(filename=vis, filename_tir=tir), img_prefix=None)

        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(pairs))

    # 解包 DataContainer
    data['img_metas'] = [m.data[0] for m in data['img_metas']]
    # img
    data['img'] = [img.data[0] if hasattr(img, 'data') else img for img in data['img']]
    # img_tir
    if 'img_tir' in data:
        data['img_tir'] = [im.data[0] if hasattr(im, 'data') else im for im in data['img_tir']]

    # GPU/CUDA scatter 或 CPU 检查
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(m, RoIPool), 'CPU inference with RoIPool is not supported currently.'

    # forward
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results