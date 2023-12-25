import torch, math
import numpy as np
from yolov6.data.data_augment import letterbox

from yolov6.core.inferer import Inferer
from typing import List, Optional
from yolov6.utils.nms import non_max_suppression

conf_thres: float =.25 #@param {type:"number"}
iou_thres: float =.45 #@param {type:"number"}
max_det:int =  1000 #@param {type:"integer"}
agnostic_nms: bool = False #@param {type:"boolean"}

class ImageProcessor:
  def __init__():
    #Set-up hardware options
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
  
  def check_img_size(img_size, s=32, floor=0):
    def make_divisible( x, divisor):
      # Upward revision the value x to make it evenly divisible by the divisor.
      return math.ceil(x / divisor) * divisor
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")
    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

  def process_image(img_src, img_size, stride, half):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0
    return image, img_src

class FrameProcessor:
  def __init__(self, img_size, stride, half, device, model):
    self.model = model
    self.img_size = img_size
    self.stride = stride
    cuda = device != 'cpu' and torch.cuda.is_available()
    self.device = torch.device('cuda:0' if cuda else 'cpu')
    if half & (device != 'cpu'):
      self.model.model.half()
    else:
      self.model.model.float()
      self.half = False
    self.model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup
                
  def process_frame(self, frame):
    detections = []
    img, img_src = ImageProcessor.process_image(frame, self.img_size, self.stride, self.half)
    img = img.to(self.device)
    if len(img.shape) == 3:
      img = img[None]
      # expand for batch dim
    pred_results = self.model(img)
    classes:Optional[List[int]] = None # the classes to keep
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(det):
      det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
      for x1, y1, x2, y2, conf, cls in reversed(det):
        detections.append([x1, y1, x2, y2, conf, cls])
    return detections