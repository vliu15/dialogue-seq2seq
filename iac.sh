#!/bin/bash

# download iacorpus dataset
if [ ! -d data/iac_v1.1 ]; then
    wget http://nldslab.soe.ucsc.edu/iac/iac_v1.1.zip && unzip iac_v1.1.zip -C data && rm iac_v1.1.zip
fi
