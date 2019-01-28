#!/bin/bash

# install dependencies
pip3 install -r requirements.txt

# moses tokenizer and bleu score
if [ ! -f tokenizer.perl ]; then 
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl; fi
if [ ! -f nonbreaking_prefix.de ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de; fi
if [ ! -f nonbreaking_prefix.en ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en; fi
sed -i '.bak' "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
if [ ! -f multi-bleu.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl; fi

# download data
if [ ! -d data/multi30k ]; then
    mkdir -p data/multi30k
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/multi30k && rm mmt16_task1_test.tar.gz
fi

# preprocess data
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i '' "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python3 preprocess.py -train_src data/multi30k/train.en.atok -train_tgt data/multi30k/train.de.atok -valid_src data/multi30k/val.en.atok -valid_tgt data/multi30k/val.de.atok -save_data data/multi30k.atok.low.pt