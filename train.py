''' This script handles the training process '''
import argparse
import math
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils.dataset import TranslationDataset, paired_collate_fn, prepare_dataloaders
from utils.metrics import cal_performance
from seq2seq import Constants
from seq2seq.Models import Seq2Seq
from seq2seq.Optim import ScheduledOptim


def train_epoch(model, training_data, optimizer, device, mmi_factor, smoothing=True):
    ''' Epoch operation in training phase '''
    model.train()   # training mode

    #- Set up logging
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    #- Iterate through batches for training
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        #- Prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        batch_size, n_steps, _ = src_seq.size()

        #- Clip the target_seq for the BOS token
        gold = tgt_seq[:, :, 1:]

        #- Forward
        optimizer.zero_grad()
        model.session.zero_lstm_state(batch_size, device)
        preds = []
        for i in range(n_steps):
            pred = model(
                src_seq[:, i, :].squeeze(1), src_pos[:, i, :].squeeze(1),
                tgt_seq[:, i, :].squeeze(1), tgt_pos[:, i, :].squeeze(1))
            preds.append(pred)

        #- Backward (use total loss)
        loss = 0
        n_correct = 0
        for i in range(n_steps):
            loss_, n_correct_ = cal_performance(preds[i], gold[:, i, :].squeeze(1), smoothing=smoothing, mmi_factor=mmi_factor)
            loss += loss_
            n_correct += n_correct_
        loss.backward()

        #- Optimizer step
        optimizer.step_and_update_lr()

        #- Logging
        total_loss += loss.item()
        n_word_correct += n_correct
        n_word_total += gold.ne(Constants.PAD).sum().item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, mmi_factor):
    ''' Epoch operation in evaluation phase '''
    model.eval()    # inference mode

    #- Set up logging
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        #- Iterate through validation batches
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            #- Prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            batch_size, n_steps, _ = src_pos.size()
            gold = tgt_seq[:, :, 1:]

            #- Reset LSTM hidden states
            model.session.zero_lstm_state(batch_size, device)
            
            #- Forward pass
            preds = []
            for i in range(n_steps):
                pred = model(
                    src_seq[:, i, :].squeeze(1), src_pos[:, i, :].squeeze(1),
                    tgt_seq[:, i, :].squeeze(1), tgt_pos[:, i, :].squeeze(1))
                preds.append(pred)

            #- Accumulate loss and accuracy
            loss = 0
            n_correct = 0
            for i in range(n_steps):
                loss_, n_correct_ = cal_performance(preds[i], gold[:, i, :].squeeze(1), smoothing=False, mmi_factor=mmi_factor)
                loss += loss_
                n_correct += n_correct_

            #- Logging
            total_loss += loss.item()
            n_word_correct += n_correct
            n_word_total += gold.ne(Constants.PAD).sum().item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt, epoch):
    ''' Start training '''
    log_train_file = None
    log_valid_file = None

    #- Prepare logs
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    #- Train and iterate through epochs 
    valid_accus = []
    for epoch_i in range(epoch, opt.epoch):
        print('[ Epoch', epoch_i, ']')

        #- Pass through training data
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, opt.mmi_factor, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'loss/word: {loss:8.5f}, elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  loss=train_loss, elapse=(time.time()-start)/60))

        #- Pass through validation data
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt.mmi_factor)
        print('  - (Validation) gppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'loss/word: {loss:8.5f}, elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    loss=valid_loss, elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        #- Prepare checkpoint
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        #- Save checkpoint
        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        #- Save logs
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=1e-2)

    parser.add_argument('-src_emb_file', type=str, default='')
    parser.add_argument('-tgt_emb_file', type=str, default='')

    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-d_hidden', type=int, default=512)
    # parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-load_model', default=None)

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-mmi_factor', type=float, default=0.0)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.d_model = opt.d_word_vec # for residual compatibility

    #- Load training and validation datasets
    data = torch.load(opt.data)
    opt.max_seq_len = data['settings'].max_seq_len
    opt.max_subseq_len = data['settings'].max_token_subseq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    #- Share src / tgt vocab weights if needed
    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    #- Initialize model
    device = torch.device('cuda' if opt.cuda else 'cpu')
    seq2seq = Seq2Seq(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_subseq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        d_hidden=opt.d_hidden,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        mmi_factor=opt.mmi_factor,
        src_emb_file=opt.src_emb_file,
        tgt_emb_file=opt.tgt_emb_file).to(device)

    #- Output total number of parameters
    model_parameters = filter(lambda p: p.requires_grad, seq2seq.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total number of parameters: {n:3.3}M'.format(n=n_params/1000000.0))

    #- Set up optimizer
    optimizer = ScheduledOptim(
        optim.Adam(filter(lambda p: p.requires_grad, seq2seq.parameters()), betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, 
        opt.n_warmup_steps, 
        lr=opt.lr
    )

    #- Load model weights from checkpoint if possible
    if opt.load_model is not None:
        checkpoint = torch.load(opt.load_model + '.chkpt')
        epoch = checkpoint['epoch']
        try:
            seq2seq.load_state_dict(checkpoint['model'])
            print('[Info] Trained model state loaded.')
        except:
            print('[Info] Model state loading failed. Checkpoint settings: {}'.format(model_opt))
            raise RuntimeError
    else:
        epoch = 0
        print('[Info] Initialized new model.')

    #- Train model
    train(seq2seq, training_data, validation_data, optimizer, device, opt, epoch + 1)

if __name__ == '__main__':
    main()
