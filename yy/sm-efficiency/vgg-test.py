import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def make_model():
    model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return model


# 特征提取
def extract_feature(model, imgpath):
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = Image.open(imgpath)  # 读取图片
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))

    tensor = img_to_tensor(img)  # 将图片转化成tensor
    tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉

    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错

    return result_npy[0]  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]


if __name__ == "__main__":
    model = make_model()
    imgpath = '1.jpg'
    tmp = extract_feature(model, imgpath)
    print(tmp.shape)  # 打印出得到的tensor的shape
    print(tmp)  # 打印出tensor的内容，其实可以换成保存tensor的语句，这里的话就留给读者自由发挥了
