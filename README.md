# Sequence-to-Sequence Generative Dialogue Systems
This is a PyTorch adaptation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" for dialogue-based memory systems. We borrow the Transformer encoder and decoder to encode input and generate response, respectively. The encoded input updates a hidden state in a simple RNN, which serves as a session memory. We train our dialogue system with the [Internet Argument Corpus v1](https://nlds.soe.ucsc.edu/iac).

## Transformer
We borrow the code for the Transformer encoder and decoder from [this](https://github.com/jadore801120/attention-is-all-you-need-pytorch) repository.

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>

## Usage
For Python2 and Python3 dependencies, see `requirements.txt`. We assume that `python` and `pip` correspond to Python2, and `python3` and `pip3` correspond to Python3.

### Setup: Data / Preprocessing
```
sh setup.sh
```
> The above command downloads and preprocesses the Internet Argument Corpus v1.1 data dataset.

### Training
```bash
python3 train.py -data data/iac/train.data.pt -save_model trained \
  -save_mode best -proj_share_weight -label_smoothing -embs_share_weight
```
> If your source and target language share one common vocabulary, use the `-embs_share_weight` flag to enable the model to share source/target word embedding.

### Testing
```bash
python3 translate.py -model trained.chkpt -vocab data/iac/train.data.pt \
  -src data/iac/test.pkl
```
