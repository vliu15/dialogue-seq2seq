#!/bin/bash

# set up dependencies manually due to mixture of python2 and python3
# make sure Python2 <= pip, python
# make sure Python3 <= pip3, python3
pip install nltk
python -c "import nltk; nltk.download('punkt')"
pip3 install torch torchvision numpy tqdm

mkdir data

# download iacorpus dataset
if [ ! -d data/iac_v1.1 ]; then
    # get dataset and code
    wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip && unzip iac_v1.1.zip -d data && rm iac_v1.1.zip
fi

# set up pickle directory
if [ ! -d data/iac ]; then
    mkdir -p data/iac
fi

# load dataset into python-loadable
if [ ! -f data/iac/*pkl ] then
    # pickle dump concise dataset
    cp load_iac.py data/iac_v1.1/code
    cd data/iac_v1.1/code && python load_iac.py

    # restructure data folder
    cd ../../ && mkdir iac
    mv iac_v1.1/*pkl iac
    cd ..
fi

# perform preprocessing
python3 preprocess.py -train_file data/iac/train.pkl -valid_file data/iac/val.pkl -save_dir data/iac -share_vocab
