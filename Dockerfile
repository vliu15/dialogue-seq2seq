FROM pytorch/pytorch:latest

#- Install system requirements
RUN apt-get update
RUN apt-get install -y git vim wget unzip
RUN apt-get install -y python python-pip python3 python3-pip

#- Point `python` to `/usr/bin/python`
RUN rm /opt/conda/bin/python
RUN alias python=/usr/bin/python2.7

#- Get code
WORKDIR /
RUN git clone https://github.com/vliu15/dialogue-seq2seq.git

#- Download data and preprocess
WORKDIR /dialogue-seq2seq
RUN pip3 install -r requirements.txt
# RUN sh setup.sh
