import torch, math
import numpy as np
from yolov6.data.data_augment import letterbox

class Yolov6Util:
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
