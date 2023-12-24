# Credits: https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/tree/master/local_env/predict_video.py
import os
import cv2
import torch
from yolov6.layers.common import DetectBackend
from yolov6.core.inferer import Inferer
from Yolov6Util import Yolov6Util
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.core.inferer import Inferer
from typing import List, Optional
import PIL
from yolov6.utils.nms import non_max_suppression

device:str = "cpu" #@param ["gpu", "cpu"]
half:bool = False #@param {type:"boolean"}
img_size:int = 640 #@param {type:"integer"}
conf_thres: float =.25 #@param {type:"number"}
iou_thres: float =.45 #@param {type:"number"}
max_det:int =  1000 #@param {type:"integer"}
agnostic_nms: bool = False #@param {type:"boolean"}
hide_labels: bool = False #@param {type:"boolean"}
hide_conf: bool = False #@param {type:"boolean"}

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'traffic.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
cap  = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model = DetectBackend(f"best_ckpt.pt", device=device)
stride = model.stride
class_names = load_yaml("dataset_config.yaml")['names']

if half & (device != 'cpu'):
  model.model.half()
else:
  model.model.float()
  half = False

if device != 'cpu':
  model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

img_size = Yolov6Util.check_img_size(img_size, s=stride)
img, img_src = Yolov6Util.process_image(frame, img_size, stride, half)

img = img.to(device)

if len(img.shape) == 3:
    img = img[None]
    # expand for batch dim
pred_results = model(img)
classes:Optional[List[int]] = None # the classes to keep
det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
img_ori = img_src.copy()
if len(det):
  det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
  for *xyxy, conf, cls in reversed(det):
      class_num = int(cls)
      label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
      Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
dest_img = PIL.Image.fromarray(img_ori)

dest_img.show()
