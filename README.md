# Adapting the Transformer for Dialogue with Memory
This is a PyTorch adaptation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" for dialogue-based memory systems. We borrow the Transformer encoder and decoder to encode input and generate response, respectively. The encoded input updates a hidden state in a simple RNN, which serves as a session memory. We train our dialogue system with the [Internet Argument Corpus v2](https://nlds.soe.ucsc.edu/iac2).

## Transformer
We borrow the code for the Transformer from [this](https://github.com/jadore801120/attention-is-all-you-need-pytorch) repository.

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>

## Base Usage: WMT'16 Multimodal Translation, Multi30k (de-en)
An example of training for the [WMT'16 Multimodal Translation task](http://www.statmt.org/wmt16/multimodal-task.html).

### Setup: Dependencies / Data
```
sh wmt.sh
``` 

### Training
```bash
python train.py -data data/multi30k.atok.low.pt -save_model trained \
  -save_mode best -proj_share_weight -label_smoothing -no_cuda
```
> If your source and target language share one common vocabulary, use the `-embs_share_weight` flag to enable the model to share source/target word embedding. 

### Testing
```bash
python translate.py -model trained.chkpt -vocab data/multi30k.atok.low.pt \
  -src data/multi30k/test.en.atok -no_cuda
```

## Training Performance

<p align="center">
<img src="https://imgur.com/rKeP1bb.png" width="400">
<img src="https://imgur.com/9je3X6U.png" width="400">
</p>

- Parameter settings:
  - default parameter and optimizer settings
  - label smoothing 
  - target embedding / pre-softmax linear layer weight sharing. 

- Elapse per epoch (on NVIDIA Titan X):
  - Training set: 0.888 minutes
  - Validation set: 0.011 minutes
