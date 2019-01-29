#!/bin/bash

# download sql dataset
if [ ! -d data/iacorpus ]; then
    mkdir -p data/iacorpus
    wget http://nldslab.soe.ucsc.edu/iac/v2/convinceme_2016_05_18.sql.gz && gunzip -c convinceme_2016_05_18.sql.gz > data/iacorpus/convinceme_2016_05_18.sql && rm convinceme_2016_05_18.sql.gz
fi
