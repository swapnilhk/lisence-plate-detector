# lisence-plate-detector

## Prerequisites
Python 3.7 - 3.10, PIP

## Training and Detection Framework
YOLOv6

## Python Virtual Environment Setup
https://docs.voxel51.com/getting_started/virtualenv.html

## Image Dataset: Google Open Images V7
https://storage.googleapis.com/openimages/web/index.html

### Dataset Format
Google Open Images V7: https://docs.ultralytics.com/datasets/detect/open-images-v7/
Yolo: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

## FiftyOne Installation
https://docs.voxel51.com/getting_started/install.html

## Download and Visualize Image using FiftyOne
https://storage.googleapis.com/openimages/web/download_v7.html#download-fiftyone

### Sample Code for Launching FiftyOne

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = fo.zoo.load_zoo_dataset(
              "open-images-v7",
              split="train",
              label_types=["detections"],
              classes=["Vehicle registration plate"],
              max_samples=100,
          )
session = fo.launch_app(dataset)
session.wait() 
```

