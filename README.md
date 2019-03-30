# Sequence-to-Sequence Generative Dialogue Systems
This is a PyTorch adaptation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" for dialogue-based memory systems. We borrow the Transformer encoder and decoder to encode input and generate response, respectively. The encoded input updates a hidden state in a simple RNN, which serves as a session memory. We train our dialogue system with the [Internet Argument Corpus v1](https://nlds.soe.ucsc.edu/iac).

## Transformer
We borrow the code for the Transformer encoder and decoder from [this](https://github.com/jadore801120/attention-is-all-you-need-pytorch) repository.

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>

## Internet Argument Corpus
The Internet Argument Corpus (IAC) is a collection of discussion posts scraped from political debate forums that we use to benchmark our model. The dataset in total has 11.8k discussions, which amount to about 390k individual posts from users. We define each example to be one discussion, a sequence of posts, which are a sequence of tokens.
> For generality, we refer to the concept of a discussion as a `seq` and a post as a `subseq`.

On the IAC dataset, we are able to to achieve ~25% word accuracy rate and a 80 perplexity score on both training and validation sets with an `<UNK>` pruning threshold in preprocessing. Without this threshold, we achieve ~28% word accuracy rate and a 66 perplexity score but at the cost of coherent and interesting output. Below, we provide some details about the default parameters we use.

- Subsequence lengths are set to 50 tokens and sequence lengths are set to 25 subsequences.
- We throw away all examples that are comprised of >5% of `<UNK>` tokens.
- We limit our vocabulary to 17k by setting a minimum word occurrence of 15.
- Fine-tuning GloVe embeddings in training adds a 1-2% boost in performance towards convergence.
- We find that a few thousand warmup steps to a learning rate around 1e-3 yields best early training.
- In general, increasing the complexity of the model does little on this task and dataset. We find that 3 Transformer encoder-decoder layers is a reasonable lower-bound.

## Usage
For Python2 and Python3 dependencies, see `requirements.txt`. We assume that `python` and `pip` correspond to Python2, and `python3` and `pip3` correspond to Python3.

### Docker
Run the following command to build and run a Docker container (without data) with all dependencies:
```bash
docker build -t seq2seq:latest .
docker run -it -v $PWD:/dialogue-seq2seq seq2seq:latest
```
> The code can be found in the `/dialogue-seq2seq` folder.


### Setup & Preprocessing
```bash
sh setup.sh
```
> Default preprocessing shares source/target vocabulary and uses GloVe pretrained embeddings.

### Training
```bash
python3 train.py -data data/iac/train.data.pt -save_model trained \
  -save_mode best -proj_share_weight -label_smoothing -embs_share_weight \
  -src_emb_file data/glove/src_emb_file.npy -tgt_emb_file data/glove/tgt_emb_file.npy
```
> Use the `-embs_share_weight` flag to enable the model to share source/target word embedding if training embeddings.

> Use the flags `-src_emb_file` and/or `-tgt_emb_file` to use pretrained embeddings.

### Testing
```bash
python3 translate.py -model trained.chkpt -test_file data/iac/test.data.pt
```

### Interactive Use
```bash
python3 interactive.py -model trained.chkpt -prepro_file data/iac/train.data.pt
```
