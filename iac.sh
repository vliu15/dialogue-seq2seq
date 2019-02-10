#!/bin/bash

# set up dependencies
pip3 install -r requirements.txt


# download iacorpus dataset
if [ ! -d data/iac_v1.1 ]; then
    # get dataset and code
    wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip && unzip -C data && rm iac_v1.1.zip
fi

# load dataset into python-loadable
if [ ! -d data/iac ]; then
    # pickle dump concise dataset
    cp load_iac.py data/iac_v1.1/code
    cd data/iac_v1.1/code && python load_iac.py

    # restructure data folder
    cd ../../ && mkdir iac
    mv iac_v1.1/*pkl iac
fi

# perform preprocessing
python3 preprocess_iac.py -train_file data/iac/train.pkl -valid_file data/iac/val.pkl -save_dir data/iac
