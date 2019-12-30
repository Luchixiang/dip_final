from __future__ import division

import os
import time

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib.ticker import NullLocator
from torch.autograd import Variable

from models import *
from utils.datasets import *
from utils.utils import *


def detect_usingYolo(img_path):

    model_def = 'config/yolov3-custom.cfg'
    img_size = 416
    weights_path = 'checkpoints/yolov3_ckpt_4_acc_0.9997410547626311.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_path = 'data/custom/classes.names'
    conf_thres = 0.3
    nms_thres = 0.4
    # os.makedirs("output_oneImg", exist_ok=True
    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode
    img = transforms.ToTensor()(Image.open(img_path))

    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, img_size)
    img = np.expand_dims(img, 0)
    print(img.shape)
    classes = load_classes(class_path)  # Extracts class labels from file

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # input_imgs = Variable(img.astype(Tensor))
    input_imgs = torch.cuda.FloatTensor(img)
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        # detections = np.squeeze(detections, axis=0)
    detections = np.array(detections)
    print(detections.shape)
    # img_detections.extend(detections)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    # Create plot
    img = cv2.imread(img_path)
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
    # cv2.namedWindow("Image_yolo")
    # cv2.imshow()
    return img

img = detect_usingYolo('data/custom/images/plate/M6_20191030084503_0_1.jpg')
print(img)