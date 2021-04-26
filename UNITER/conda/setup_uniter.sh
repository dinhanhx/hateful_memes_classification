conda create -n uniter python=3.6 pip --yes
conda activate uniter

cd ../

conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.2 -c pytorch --yes
conda install scikit-learn --yes
pip install pytorch-pretrained-bert==0.6.2 tensorboardX==1.7 ipdb==0.12 lz4==2.1.9 lmdb==0.97
pip install toolz cytoolz msgpack msgpack-numpy

HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod

# apex (gcc >= 5.4.0 required)
git clone https://github.com/jackroos/apex
cd ./apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
