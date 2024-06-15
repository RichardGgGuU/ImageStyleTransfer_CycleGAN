import torch

# 定义权重文件的绝对路径
weight_path = 'C:/Users/28617/PycharmProjects/ImageStyleTransfer_CycleGAN/weights/monet2photo.pth'

# 加载检查点文件
ckpt = torch.load(weight_path)

# 打印检查点文件的键，检查是否包含 'Ga_model'
print("Keys in checkpoint file:", ckpt.keys())

# 检查 'Ga_model' 是否存在于检查点文件中
if 'Ga_model' in ckpt:
    Cycle_G_A.load_state_dict(ckpt['Ga_model'], strict=False)
else:
    print("Error: 'Ga_model' key not found in checkpoint file")
    # 处理缺少键的情况，可以根据需要添加逻辑
