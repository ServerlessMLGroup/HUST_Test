import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image

if __name__ == "__main__":
    densenet = densenet121(pretrained=True)
    densenet.eval()

    img = Image.open("1.jpg")

    trans_ops = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    images = trans_ops(img).view(-1, 3, 224, 224)
    outputs = densenet(images)

    _, predictions = outputs.topk(5, dim=1)

    labels = list(map(lambda s: s.strip(), open("./data/imagenet/synset_words.txt").readlines()))
    for idx in predictions.numpy()[0]:
        print("Predicted labels:", labels[idx])