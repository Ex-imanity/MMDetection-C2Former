from mmdet.apis import init_detector
from mmdet.apis.inference import inference_detector_paired  # 新增的接口
import mmcv

# 双模态模型配置与权重
config_file = 'configs/cascade_rcnn/cascade_rcnn_c2former_fpn_dconv_c3-c5_2x_drone_vehicle.py'
checkpoint_file = 'work_dirs/cascade_rcnn_c2former_fpn_dconv_c3-c5_2x_drone_vehicle/epoch_22.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 方式一：文件路径输入（推荐，走 LoadPairedImageFromFile）
vis_path = '/home/hclserver/zp/Data/VOC2012/JPEGImages/test_00127.jpg'
tir_path = '/home/hclserver/zp/Data/VOC2012/JPEGImages/test_00127_tir.jpg'
result = inference_detector_paired(model, (vis_path, tir_path))

# 可视化到两张图（结果一致，绘制到两模态）
model.show_result(vis_path, result, out_file='work_dirs/infer_paired/vis_result.jpg')
model.show_result(tir_path, result, out_file='work_dirs/infer_paired/tir_result.jpg')

# # 方式二：内存数组输入（不走文件加载）
# vis_img = mmcv.imread(vis_path)
# tir_img = mmcv.imread(tir_path)
# result2 = inference_detector_paired(model, (vis_img, tir_img))

# # 批量输入（列表中每个元素都是一对）
# pairs = [(vis_path, tir_path),
#          ('path/to/000124.jpg', 'path/to/000124_tir.jpg')]
# batch_results = inference_detector_paired(model, pairs)