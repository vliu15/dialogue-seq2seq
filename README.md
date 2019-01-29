# Adapting the Transformer for Dialogue with Memory

This is a PyTorch adaptation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>

## Usage: WMT'16 Multimodal Translation, Multi30k (de-en)
An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

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

## Acknowledgements
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert and @ZiJianZhao.
