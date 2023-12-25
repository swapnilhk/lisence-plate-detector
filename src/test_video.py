import os
import cv2
from yolov6.layers.common import DetectBackend
from Yolov6Util import FrameProcessor,  ImageProcessor
from yolov6.utils.events import load_yaml

device:str = "cpu" #@param ["gpu", "cpu"]
half:bool = False #@param {type:"boolean"}
img_size:int = 640 #@param {type:"integer"}

VIDEOS_DIR = os.path.join('/Users/sw20039189/Movies/', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'traffic.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
cap  = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

class_names = load_yaml("dataset_config.yaml")['names']

model = DetectBackend(f"license_plate_detector_ckpt.pt", device=device)
frameProcessor = FrameProcessor(img_size, half,  device, model)

while ret:
  for x1, y1, x2, y2, conf, cls in frameProcessor.process_frame(frame):
    if float(conf) > 0.6:
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
      class_num = int(cls)
      cv2.putText(frame, class_names[class_num].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
  out.write(frame)
  ret, frame = cap.read()
