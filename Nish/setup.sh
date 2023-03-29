apt-get update
apt-get install python3.10-dev libgl1-mesa-glx unzip python3-setuptools vim --yes
pip install transformers opencv-python scikit-learn datasets ipdb --quiet
python -m pip install --upgrade pip
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# if [ ! -d "tiny-imagenet-200" ]
# then
#     wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
#     unzip tiny-imagenet-200.zip
#     rm tiny-imagenet-200.zip
# fi
