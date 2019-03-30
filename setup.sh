#!/bin/bash

#- Set up dependencies where python2->Python2 and python3->Python3
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt')"

#- Download IACorpus dataset
if [ ! -d data/iac_v1.1 ]; then
    #- Get dataset and code
    wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip && unzip iac_v1.1.zip -d data && rm iac_v1.1.zip
fi

mkdir -p data/iac

#- Load dataset into python-loadable
if [ ! -f data/iac/test.pkl ]; then
    #- Pickle dump concise dataset
    cp load_iac.py data/iac_v1.1/code
    cd data/iac_v1.1/code && python2 load_iac.py

    #- Restructure data folder
    cd ../../ && mv iac_v1.1/*pkl iac
    cd ..
fi

mkdir -p data/glove

#- Download glove dataset
if [ ! -f data/glove/glove.6B.300d.txt ]; then
    #- Get dataset
    wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip -d data/glove && rm glove.6B.zip
fi

#- Perform preprocessing
python3 preprocess.py -train_file data/iac/train.1.pkl,data/iac/train.2.pkl,data/iac/train.3.pkl,data/iac/train.4.pkl \
    -valid_file data/iac/val.pkl -test_file data/iac/test.pkl -save_dir data/iac -share_vocab -use_glove_emb
