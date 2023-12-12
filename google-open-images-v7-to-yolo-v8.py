import os
import yaml
import re
from pathlib import Path
import shutil

SRC_DATASET_DIR = '/Users/sw20039189/fiftyone/open-images-v7'
DEST_DATASET_DIR = SRC_DATASET_DIR+'/yolov8'

# remove given trailing character if it exists
def removeTrailingCharacter(str, trailing_char_to_be_removed):
   if(str[-1]==trailing_char_to_be_removed):
     return str[:-1] 

# load dest dataset config
with open('dataset_config.yaml', 'r') as file:
  dataset_config = yaml.safe_load(file)

# remove previous yolo format data if it exiest
if os.path.isdir(DEST_DATASET_DIR):
  shutil.rmtree(DEST_DATASET_DIR)

# create dest folder structure
# create hard links
Path(dataset_config['path']+'/labels/train/').mkdir(parents=True, exist_ok=True)
Path(dataset_config['path']+'/labels/val/').mkdir(parents=True, exist_ok=True)
Path(dataset_config['path']+'/labels/test/').mkdir(parents=True, exist_ok=True)
# create directories
shutil.copytree(SRC_DATASET_DIR + '/train/data', dataset_config['path']+'/'+dataset_config['train'], copy_function=os.link)
if os.path.isdir(SRC_DATASET_DIR + '/val/data'):
  shutil.copytree(SRC_DATASET_DIR + '/val/data', dataset_config['path']+'/'+dataset_config['val'], copy_function=os.link)
if os.path.isdir(SRC_DATASET_DIR + '/test/data'):
  shutil.copytree(SRC_DATASET_DIR + '/test/data', dataset_config['path']+'/'+dataset_config['test'], copy_function=os.link)

# find image ids in datast
dataset_file_list = os.scandir(SRC_DATASET_DIR + '/train/data')
dataset_images_ids=[]
for entry in dataset_file_list:
  dataset_images_ids.append(entry.name[:-4]) # remove file extension

# invert the 'names' map in yolo dataset config
inverted_yolov8_class_map={}
# create regex to extract LabelNames from metadate file
metadata_expression  = ''
for key, value in dataset_config['names'].items():
   inverted_yolov8_class_map[value]=key
   metadata_expression += value + '|'
metadata_expression  =  removeTrailingCharacter(metadata_expression, '|')  # remove trailing '|' if it exists
metadata_expression = '(.+),(' + metadata_expression + ')'

# map containing google open api v7 LabelNames to yolov8 class indexs
open_images_lable_name_to_yolo_class_index={}
# create regex to extract information from open images detections.csv file
detections_label_name_subexpression = ''
with open(SRC_DATASET_DIR + '/train/metadata/classes.csv', 'r') as open_images_v7_metadata:
  for line in open_images_v7_metadata:
        match = re.search(metadata_expression, line)
        if match:
           open_images_lable_name_to_yolo_class_index[match.group(1)]=inverted_yolov8_class_map[match.group(2)]
           detections_label_name_subexpression += match.group(1) + '|'
  detections_label_name_subexpression  =  removeTrailingCharacter(detections_label_name_subexpression, '|')  # remove trailing '|' if it exists

detections_expression = ''
for dataset_images_id in dataset_images_ids:
   detections_expression += dataset_images_id + '|'
detections_expression  =  removeTrailingCharacter(detections_expression, '|')  # remove trailing '|' if it exists
detections_expression = '(' + detections_expression +  '),\w+,('+detections_label_name_subexpression+'),[\d.],(0\.\d+),(0\.\d+),(0\.\d+),(0\.\d+)'

# convert data to yolov8 format
with open(SRC_DATASET_DIR + 'train/labels/detections.csv', 'r') as open_images_v7_detections:
  for line in open_images_v7_detections:
    match = re.search(detections_expression, line)    
    if match:
      print(match.group(0))
      normalised_x = (float(match.group(3)) + float(match.group(4))) / 2
      normalised_y = (float(match.group(5)) + float(match.group(6))) / 2
      normalised_w = float(match.group(4)) - float(match.group(3))
      normalised_h = float(match.group(6)) - float(match.group(5))
      with open(dataset_config['path']+'/labels/train/'+match.group(1)+'.txt', 'a') as yolov8_detections:
         yolov8_detections.write(str(open_images_lable_name_to_yolo_class_index[match.group(2)])
          + ' ' + str(normalised_x)
          + ' ' + str(normalised_y)
          + ' ' + str(normalised_w)
          + ' ' + str(normalised_h)
          + '\n')
      