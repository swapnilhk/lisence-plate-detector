import os
import yaml
import re
from pathlib import Path
import shutil

def removeTrailingCharacter(str, trailing_char_to_be_removed):
   if(str[-1]==trailing_char_to_be_removed):
     return str[:-1]  # remove trailing '|' if it exists

with open('dataset_config.yaml', 'r') as file:
  dataset_config = yaml.safe_load(file)

# find image ids in datast
dataset_file_list = os.scandir(dataset_config['path']+'/'+dataset_config['train'])

dataset_images_ids=[]
for entry in dataset_file_list:
  dataset_images_ids.append(entry.name[:-4]) # remove file extension

# map  yolov8 class indexs to google open api v7 LabelNames
inverted_yolov8_class_map={}
metadata_expression  = ''
for key, value in dataset_config['names'].items():
   inverted_yolov8_class_map[value]=key
   metadata_expression += value + '|'
metadata_expression  =  removeTrailingCharacter(metadata_expression, '|')  # remove trailing '|' if it exists
metadata_expression = '(.+),(' + metadata_expression + ')'

open_images_lable_name_to_yolo_class_index={}
detections_label_name_subexpression = ''
with open(dataset_config['path']+'/train/metadata/classes.csv', 'r') as open_images_v7_metadata:
  for line in open_images_v7_metadata:
        match = re.search(metadata_expression, line)
        if match:
           open_images_lable_name_to_yolo_class_index[match.group(1)]=inverted_yolov8_class_map[match.group(2)]
           detections_label_name_subexpression += match.group(1) + '|'
  detections_label_name_subexpression  =  removeTrailingCharacter(detections_label_name_subexpression, '|')  # remove trailing '|' if it exists

# convert data to yolov8 format
detections_expression = ''
for dataset_images_id in dataset_images_ids:
   detections_expression += dataset_images_id + '|'
detections_expression  =  removeTrailingCharacter(detections_expression, '|')  # remove trailing '|' if it exists

detections_expression = '(' + detections_expression +  '),\w+,('+detections_label_name_subexpression+'),[\d.],(0\.\d+),(0\.\d+),(0\.\d+),(0\.\d+)'

# remove previous data
if os.path.isdir(dataset_config['path']+'/train/labels/detections_yolov8'):
  shutil.rmtree(dataset_config['path']+'/train/labels/detections_yolov8')

Path(dataset_config['path']+'/train/labels/detections_yolov8').mkdir(parents=True, exist_ok=True)
with open(dataset_config['path']+'/train/labels/detections.csv', 'r') as open_images_v7_detections:
  for line in open_images_v7_detections:
    match = re.search(detections_expression, line)    
    if match:
      print(match.group(0))
      normalised_x = (float(match.group(3)) + float(match.group(4))) / 2
      normalised_y = (float(match.group(5)) + float(match.group(6))) / 2
      normalised_w = float(match.group(4)) - float(match.group(3))
      normalised_h = float(match.group(6)) - float(match.group(5))
      with open(dataset_config['path']+'/train/labels/detections_yolov8/'+match.group(1)+'.txt', 'a') as yolov8_detections:
         yolov8_detections.write(str(open_images_lable_name_to_yolo_class_index[match.group(2)])
          + ' ' + str(normalised_x)
          + ' ' + str(normalised_y)
          + ' ' + str(normalised_w)
          + ' ' + str(normalised_h)
          + '\n')
      