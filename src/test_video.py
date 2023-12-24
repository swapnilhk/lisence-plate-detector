import os
import cv2
import torch
from yolov6.layers.common import DetectBackend
from yolov6.core.inferer import Inferer
from Yolov6Util import Yolov6Util
from yolov6.utils.events import load_yaml
from yolov6.core.inferer import Inferer
from typing import List, Optional
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

VIDEOS_DIR = os.path.join('/Users/sw20039189/Movies/', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'traffic.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
cap  = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model = DetectBackend(f"license_plate_detector_ckpt.pt", device=device)
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

while ret:
  img, img_src = Yolov6Util.process_image(frame, img_size, stride, half)

  img = img.to(device)

  if len(img.shape) == 3:
      img = img[None]
      # expand for batch dim
  pred_results = model(img)
  classes:Optional[List[int]] = None # the classes to keep
  det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

  if len(det):
    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
    for x1, y1, x2, y2, conf, cls in reversed(det):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        class_num = int(cls)
        cv2.putText(frame, class_names[class_num].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
  out.write(frame)
  ret, frame = cap.read()
