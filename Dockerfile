FROM ubuntu:18.04

#- Prepare ubuntu system
RUN apt-get update
RUN apt-get install -y wget unzip vim
RUN apt-get install -y aptitude
RUN aptitude install -y git 
RUN apt-get install -y python python3 python-pip python3-pip

#- Install Python dependencies
RUN pip install nltk
RUN python -c "import nltk; nltk.download('punkt')"
RUN pip3 install nltk torch torchvision tqdm numpy
RUN python3 -c "import nltk; nltk.download('punkt')"

#- Prepare repository
WORKDIR /
RUN git clone https://github.com/vliu15/transformer-rnn-pytorch.git
WORKDIR /transformer-rnn-pytorch

#- Download & preprocess data
RUN sh setup.sh