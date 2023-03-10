import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from glob import glob
from torchvision import transforms
import time
from torchvision import datasets, models, transforms
from torchvision.io import read_image
import sys

from myutils.dataset import letterbox


if __name__ == '__main__':
    # args parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_times', type = int, default = 0)
    parser.add_argument('--resnet_times', type = int, default = 1)
    parser.add_argument('--cudnn', type = str, default = "True")
    parser.add_argument('--cuda_device', type = int, default = 3)
    parser.add_argument('--yolo_batch_size', type = int, default = 1)
    parser.add_argument('--resnet_batch_size', type = int, default = 1)
    args = parser.parse_args()

    #torch.cuda.set_device(args.cuda_device) 
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.cuda_device
    device = torch.device("cuda:%d" % args.cuda_device if torch.cuda.is_available() else "cpu")

    if args.cudnn == "True":
        print("cudnn is on!")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        print("cudnn is off!")
    # yolo stage

    # Model

    # yolo_model = torch.hub.load('/root/.cache/torch/hub/ultralytics_yolov5_master', 'yolov5s', source = "local", device = device)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device)

    # Images
    yolo_img_paths = glob('{:s}/*'.format('/workspace/pytorch/jpgimgs/jpgs'))
    imgs = []  # batch of images
    # multi inference
    for img_path in yolo_img_paths:
        img0 = cv2.imread(img_path)  # PIL image
        #img = resize(img)
        imgs.append(img0)

    # # multi inference
    # timing = []
    # for i in range(1, 100 + 1):
    #     img_list = imgs[0:args.yolo_batch_size]
    #     T0 = time.perf_counter()
    #     for img in img_list:
    #         img = letterbox(img, 640, stride=32, auto=True)[0]
    #         # Convert
    #         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #         img = np.ascontiguousarray(img)
    #         #yolo_model(img) #no error!
    #     T1 = time.perf_counter()
    #     timing.append(T1 - T0)
    #     if i % 10 == 0:
    #         print('Iteration %d/%d, ave batch inference time %.2f ms'%(i, 100, np.mean(timing) * 1000))
    
    results = yolo_model(imgs, size = 640)

    crops = results.crop(save=True)                
               
    # os.system("mkdir /workspace/pytorch/runs/detect/exp/crop")
    # os.system("mv  /workspace/pytorch/runs/detect/exp/crops/person /workspace/pytorch/runs/detect/exp/crop")

    # resnet stage

    # model = torch.hub.load('/root/.cache/torch/hub/pytorch_vision_v0.10.0', 'resnet152', source = "local", pretrained=True)

    # preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # model = model.eval().to(device)

    # resnet_img_paths = glob('{:s}/*'.format('/workspace/pytorch/runs/detect/exp/crop/person'))
    # #dataset = {'predict' : datasets.ImageFolder("/workspace/pytorch/runs/detect/exp/crop", preprocess)}
    # #dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = args.resnet_batch_size, shuffle=True, num_workers=0)}
    # # multi resnet
    # input_shape=(args.resnet_batch_size, 3, 224, 224)
    # input_data_warm = torch.randn(input_shape)
    # input_data_warm = input_data_warm.to(device)
    # # warm up
    # print("Warming up model...")
    # with torch.no_grad():
    #     for _ in range(50):
    #         __ = model(input_data_warm)
    # torch.cuda.synchronize()
    # print("Start timing ...")
    # all = 0
    # timings = []
    # in_timings = []
    # with torch.no_grad():
    #     times = 0
    #     preprocess_time = 0
    #     for img_path in resnet_img_paths:
    #         all += 1
    #         input_image = Image.open(img_path)
    #         T0 = time.perf_counter()
    #         input_tensor = preprocess(input_image)
    #         if times == 0 :
    #             input_batch = input_tensor.unsqueeze(0)
    #         else:
    #             input_batch = torch.cat((input_batch, input_tensor.unsqueeze(0)), dim=0) # GPU?
    #         T1 = time.perf_counter()
    #         preprocess_time += T1 - T0
    #         times += 1
    #         if times % args.resnet_batch_size == 0:
    #             timings.append(preprocess_time)
    #             times = 0
    #             preprocess_time = 0
    #             inputs = input_batch.to(device)
    #             for ii in range(10):
    #                 _ = model(inputs)
    #             torch.cuda.synchronize()
    #             T0 = time.perf_counter()
    #             output = model(inputs)
    #             T1 = time.perf_counter()
    #             in_timings.append(T1 - T0)
    # print(len(timings))
    # print("resnet avg preprocess cost(per batch without img open):%s ms" % ((np.mean(timings) * 1000)))
    # #print("resnet avg inference cost(per batch):%s ms" % ((np.mean(in_timings) * 1000)))

    # os.system("rm -r /workspace/pytorch/runs/detect")