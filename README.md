# Sequence-to-Sequence Generative Dialogue Systems
This is a Pytorch adaptation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" for memory-based generative dialogue systems. We borrow the Transformer encoder and decoder to encode decode individual responses. The encoded input updates a hidden state in an LSTM, which serves as a session memory. We train our dialogue system with the "[Internet Argument Corpus v1](https://nlds.soe.ucsc.edu/iac)".

## Seq2Seq Architecture
We adopt a hierarchical architecture, where the higher level consists of an LSTM that updates its hidden state with every input, and the lower level consists of Transformer encoder and decoder blocks to process and generate individual responses.

### Local Attention: Transformer
We adapt the code for the Transformer encoder and decoder from [this](https://github.com/jadore801120/attention-is-all-you-need-pytorch) repository.

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>

### Global Attention: Session Memory
Because of the Transformer encoder exchanges a sequence of hidden states with the decoder, we must infuse global session information by means of a single hidden state (compatible with an LSTM) while maintaining the shape of the encoder output. To do so, we design the following forward pass:

1. Max-pool across the time axis to extract prominent features.
2. Take one step through the LSTM with this feature vector.
3. Compute attention between the updated LSTM state and each position in the encoder output.
4. Softmax the attention weights to get the attention distribution.
5. Weight the encoder output accordingly and apply layer normalization to the residual connection.
6. Feed this quantity into the decoder.

## Internet Argument Corpus
The Internet Argument Corpus (IAC) is a collection of discussion posts scraped from political debate forums that we use to benchmark our model. The dataset in total has 11.8k discussions, which amount to about 390k individual posts from users. We define each example to be one discussion, a sequence of posts, which are a sequence of tokens.
> For generality, we refer to the concept of a discussion as a `seq` and a post as a `subseq`.

We believe we compete with the generative system proposed in the "[Dave the Debater](https://aclweb.org/anthology/W18-5215)" paper that won IBM Best Paper Award in 2018. The generative system described in this paper achieves a perplexity of around 70-80 and generates mediocre responses at best. Their interactive web demo can be found [here](http://114.212.80.16:8000/debate/).

### Results
On the IAC dataset, we are able to to achieve ~26% word accuracy rate and a 76 perplexity score on both training and validation sets with an `<UNK>` pruning threshold in preprocessing. Without this threshold, we achieve ~28% word accuracy rate and a 66 perplexity score but at the cost of coherent and interesting output. Below, we provide some details about the default parameters we use.

- Subsequence lengths are set to 50 tokens and sequence lengths are set to 25 subsequences.
- We throw away all examples that are comprised of >7.5% of `<UNK>` tokens.
- We limit our vocabulary to 21k by setting a minimum word occurrence of 10.
- Fine-tuning GloVe embeddings in training adds a 1-2% boost in performance towards convergence.
- We find that a few thousand warmup steps to a learning rate around 1e-3 yields best early training. We remove learning rate annealing for faster convergence.
- In general, increasing the complexity of the model does little on this task and dataset. We find that 3 Transformer encoder-decoder layers is a reasonable lower-bound.
- We find that training with the MLE objective instead of the MMI objective with cross entropy loss yields stabler training.
- For faster convergence, we adopt two phases of pretraining to familiarize the model with language modeling: denoising the autoencoder by training it to predict its input sequence, and pair prediction, where each subsequence pair is a training instance.

## Usage
For Python3 dependencies, see `requirements.txt`. For consistency, `python2` and `pip2` correspond to Python2, and `python` and `pip` correspond to Python3.

### Docker
Run the following command to build and run a Docker container (without data) with all dependencies:
```bash
docker build -t seq2seq:latest .
docker run -it -v $PWD:/dialogue-seq2seq seq2seq:latest
```
> The code can be found in the `/dialogue-seq2seq` folder.

### Setup & Preprocessing
```bash
pip install -r requirements.txt
python -m spacy download en
sh setup.sh
```
> Default preprocessing shares source/target vocabulary and uses GloVe pretrained embeddings.

### Training
```bash
python train.py -data data/iac/train.data.pt -save_model seq2seq -log seq2seq \
  -save_mode best -proj_share_weight -label_smoothing -embs_share_weight \
  -src_emb_file data/glove/src_emb_file.npy -tgt_emb_file data/glove/tgt_emb_file.npy
```
> Use the `-embs_share_weight` flag to enable the model to share source/target word embedding if training embeddings.

> Use the flags `-src_emb_file` and/or `-tgt_emb_file` to use pretrained embeddings.

### Testing
```bash
python test.py -model seq2seq.chkpt -test_file data/iac/test.data.pt
```

### Interactive Use
```bash
python interactive.py -model seq2seq.chkpt -prepro_file data/iac/train.data.pt
```
