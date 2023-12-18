echo ">>>>>>>>>>  Installing prerequisites"
sudo apt-get -y update
sudo apt-get -y install git
sudo apt-get -y install python3.10
sudo apt-get -y install python3-pip

echo ">>>>>>>>>> Cloning lisence-plate-detector"
git clone git@github.com:swapnilhk/lisence-plate-detector

echo ">>>>>>>>>> Getting dataset"
python3 -m venv env
cd env
. env/bin/activate
pip install --upgrade pip setuptools wheel
pip install fiftyone
cp  ../lisence-plate-detector/download_dataset.py .
python download_dataset.py
deactivate

echo ">>>>>>>>>> Converting dataset to yolo format"
cd ../lisence-plate-detector
python3 convert_google_open_images_v7_to_yolo.py

echo ">>>>>>>>>> Training data using Yolov6"
cd ..
git clone git@github.com:meituan/YOLOv6.git
cd YOLOv6
pip3 install -r requirements.txt

python3 tools/train.py --batch 32 --conf configs/yolov6s_finetune.py --data ../lisence-plate-detector/dataset_config.yaml --fuse_ab --device 0
