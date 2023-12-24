echo ">>>>>>>>>>  Installing prerequisites"
sudo apt-get -y update
sudo apt-get -y install git
sudo apt-get -y install python3.10
sudo apt-get -y install python3-pip
sudo apt-get -y install python3-venv
echo ">>>>>>>>>> Cloning lisence-plate-detector"
ssh-keyscan "github.com" >> ~/.ssh/known_hosts
git clone https://github.com/swapnilhk/lisence-plate-detector
cd lisence-plate-detector
git checkout implementation-using-yolov6
git pull origin implementation-using-yolov6
cd ..

echo ">>>>>>>>>> Getting dataset"
mkdir env
cd env
python3 -m venv env
. env/bin/activate
pip install --upgrade pip setuptools wheel
pip install fiftyone
cp  ../lisence-plate-detector/download_dataset.py .
python download_dataset.py
deactivate

echo ">>>>>>>>>> Converting dataset to yolo format"
cd ../lisence-plate-detector
pip3 install -r requirements.txt
python3 convert_google_open_images_v7_to_yolo.py

echo ">>>>>>>>>> Training data using Yolov6"
cd ..
git clone https://github.com/meituan/YOLOv6.git
cd YOLOv6
pip3 install -r requirements.txt

cd /usr/local/cuda/lib64
sudo rm -f libcudnn*
cd /usr/local/cuda/include
sudo rm -f cudnn*

# Train
python3 tools/train.py --batch 16 --conf configs/yolov6s_finetune.py --data ~/dataset_config.yaml --fuse_ab --device 0 --epochs 10
# Validate
python3 tools/eval.py --data ~/dataset_config.yaml --weights runs/train/exp7/weights/best_ckpt.pt --task val --device 0
# Tets
python3 tools/infer.py --weights "runs/train/exp7/weights/best_ckpt.pt" --source .003a5aaf6d17c917.jpg --yaml ~/dataset_config.yaml --device 0
