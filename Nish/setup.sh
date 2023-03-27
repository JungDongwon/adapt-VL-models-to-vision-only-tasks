apt-get update
apt-get install python3.10-dev libgl1-mesa-glx unzip vim --yes
pip install transformers opencv-python --quiet
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
