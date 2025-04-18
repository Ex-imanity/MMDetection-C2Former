
# C<sup>2</sup>Former on MMdetection

本仓库为将[C2Former]([yuanmaoxun/C2Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection](https://github.com/yuanmaoxun/C2Former))迁移到[MMDetection]([open-mmlab/mmdetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection))上的非官方实现，主要基于Cascade R-CNN和M3FD数据集。如有错误欢迎指正。



## Results

### 可见光单模态

<p align="center">
  <img src="resources\m3fd_voc_cascade_rcnn_vi_1x.png" alt="results" width="90%">
</p>

### 红外光单模态

<p align="center">
  <img src="resources\m3fd_voc_cascade_rcnn_ir_1x.png" alt="results" width="90%">
</p>

### BDLfusion后单模态

<p align="center">
  <img src="resources\m3fd_voc_cascade_rcnn_fusion_1x.png" alt="results" width="90%">
</p>

### 双模态

<p align="center">
  <img src="resources\m3fd_voc_cascade_rcnn_c2former_1x.png" alt="results" width="90%">
</p>

**注：BDLfusion请参考[LiuZhu-CV/BDLFusion: Bi-level Dynamic Learning for Jointly Multi-modality Image Fusion and Beyond (IJCAI 23)](https://github.com/LiuZhu-CV/BDLFusion)**

### Checkpoints

[百度云](https://pan.baidu.com/s/1EfkyGTk6ylNAvMGGwuZULQ?pwd=dgpp)

提取码：dgpp

## Installation 

**参照[MMdetection2.28.2](https://mmdetection.readthedocs.io/en/v2.28.2/)，由于从MMrotate迁移过来，mmdet为2.28版本，注意不是最新版！**

### Dataset

[百度云](https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD)

数据集参考文献：[JinyuanLiu-CV/TarDAL: CVPR 2022 | Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection.](https://github.com/JinyuanLiu-CV/TarDAL)

## Getting Started

### Train with a single GPU. 

```shell
python tools/train.py configs/cascade_rcnn/cascade_rcnn_c2former_fpn_1x_m3fd.py --work-dir work_dirs/cascade_rcnn_c2former_fpn_1x_m3fd
```

### Notes

数据集组织格式以及evaluation metrics为Pascal VOC格式，可见光和红外光图片放在同一文件夹/JPEGImages下，以后缀名"\*.png"和"\*tir.png"区分。由于M3FD两种模态的标注没有区分，故/Annotations文件夹中的标注以及/ImageSets/Main文件夹中的训练集验证集划分不用做模态层面的区分。

其它注意事项，例如对于多模态的输入和mmdet中detector的代码层面修改等，有空会再后续补充。建议仔细对比MMDetection原始项目和参照官方文档。
