#!/usr/local/bin/python3
from tqdm import tqdm
import transformer.Constants as Constants
import argparse
import pickle
import torch
import numpy as np
from load_glove import create_glove_emb_table

def process_sequence(seq, max_post_len, max_disc_len, keep_case):
    ''' Trim to max lengths '''
    # track trimmed counts for warnings
    trimmed_disc_count = 0
    trimmed_post_count = 0
    # trim discussion lengths to max
    if len(seq) > max_disc_len:
        seq = seq[:max_disc_len]
        trimmed_disc_count += 1
    # trim post lengths to max
    for i, post in enumerate(seq):
        tmp = post
        if len(tmp) > max_post_len:
            tmp = tmp[:max_post_len]
            trimmed_post_count += 1
        # lowercase normalization if specified
        if not keep_case:
            tmp = [word.lower() for word in tmp]
        if tmp:
            seq[i] = [Constants.BOS_WORD] + tmp + [Constants.EOS_WORD]
        else:
            seq[i] = None

    return seq, trimmed_disc_count, trimmed_post_count

def read_instances(inst, max_post_len, max_disc_len, keep_case, split_name):
    ''' Each inst is a dataset in the following format:
        [
            {
                'src': [...],
                'tgt': [...]
            },
            ...
        ]
    '''
    # generate all src and tgt insts
    src_insts = []
    tgt_insts = []
    # log counts of trimmed sequences
    trimmed_disc_count_src = 0
    trimmed_post_count_src = 0
    trimmed_disc_count_tgt = 0
    trimmed_post_count_tgt = 0
    # iterate through each dictionary
    for disc in tqdm(inst):
        src_inst, tdcs, tpcs = process_sequence(disc['src'], max_post_len, max_disc_len, keep_case)
        tgt_inst, tdct, tpct = process_sequence(disc['tgt'], max_post_len, max_disc_len, keep_case)

        src_insts.append(src_inst)
        tgt_insts.append(tgt_inst)

        trimmed_disc_count_src += tdcs
        trimmed_post_count_src += tpcs
        trimmed_disc_count_tgt += tdct
        trimmed_post_count_tgt += tpct


    print('[Info] Get {} instances from {}'.format(len(src_insts), split_name + '-src'))
    print('[Info] Get {} instances from {}'.format(len(tgt_insts), split_name + '-tgt'))

    if trimmed_disc_count_src > 0:
        print('[Warning] {}: {} instances are trimmed to the max discussion length {}'
            .format(split_name + '-src', trimmed_disc_count_src, max_disc_len))
    if trimmed_post_count_src > 0:
        print('[Warning] {}: {} subinstances are trimmed to the max post length {}'
            .format(split_name + '-src', trimmed_post_count_src, max_post_len))
    if trimmed_disc_count_tgt > 0:
        print('[Warning] {}: {} instances are trimmed to the max discussion length {}'
            .format(split_name + '-tgt', trimmed_disc_count_tgt, max_disc_len))
    if trimmed_post_count_tgt > 0:
        print('[Warning] {}: {} subinstances are trimmed to the max post length {}'
            .format(split_name + '-tgt', trimmed_post_count_tgt, max_post_len))

    return src_insts, tgt_insts

def prune(src_word_insts, tgt_word_insts, split_name):
    # check that there are same number of src/tgt instances
    if len(src_word_insts) != len(tgt_word_insts):
        print('[Warning] The {} instance count is not equal.'.format(split_name))
        min_inst_count = min(len(src_word_insts), len(tgt_word_insts))
        src_word_insts = src_word_insts[:min_inst_count]
        tgt_word_insts = tgt_word_insts[:min_inst_count]

    # check that each instances has same number of src/tgt sequences
    mismatch_count = 0
    for idx in range(len(tgt_word_insts)):
        s = src_word_insts[idx]
        t = tgt_word_insts[idx]
        if len(s) != len(t):
            min_seq_count = min(len(s), len(t))
            src_word_insts[idx] = s[:min_seq_count]
            tgt_word_insts[idx] = t[:min_seq_count]
            mismatch_count += 1
    
    if mismatch_count > 0:
        print('[Warning] There are {} mismatches in {} sequences.'.format(mismatch_count, split_name))

    # filter empty instances and sequences
    empty_count = 0
    src, tgt = [], []
    # iterate per instance
    for src_inst, tgt_inst in zip(src_word_insts, tgt_word_insts):
        s, t = [], []
        # iterate per sequence
        for src_seq, tgt_seq in zip(src_inst, tgt_inst):
            if src_seq and tgt_seq:
                s.append(src_seq)
                t.append(tgt_seq)
        if s and t:
            src.append(s)
            tgt.append(t)
        else:
            empty_count += 1
    if empty_count > 0:
        print('[Info] There are {} empty sequences.'.format(empty_count))

    return src, tgt

def build_vocab_idx(word_insts, min_word_count):
    ''' Generate vocabulary given minimum count threshold '''

    full_vocab = set([w for thread in word_insts for seq in thread for w in seq])
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for disc in word_insts:
        for seq in disc:
            for w in seq:
                word_count[w] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count >= min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print('[Info] Ignored word count = {}'.format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx, unk_prop_max, split_name):
    ''' Map words to idx sequence '''
    def check_unk_prop(d):
        # p is a post of word indices
        count_unks = lambda p: sum(1 for t in p if t == Constants.UNK)

        num_unks = float(sum(count_unks(p) for p in d))
        num_toks = float(sum(len(p) for p in d))
        prop = num_unks / num_toks <= unk_prop_max
        return prop

    above_max_count = 0
    src_idx_insts, tgt_idx_insts = [], []
    for src_disc, tgt_disc in word_insts:
        src_idx = [[word2idx.get(word, Constants.UNK) for word in post] for post in src_disc]
        tgt_idx = [[word2idx.get(word, Constants.UNK) for word in post] for post in tgt_disc]
        if check_unk_prop(src_idx) and check_unk_prop(tgt_idx):
            src_idx_insts.append(src_idx)
            tgt_idx_insts.append(tgt_idx)
        else:
            above_max_count += 1

    print('[Info] Number of {} examples above <unk> threshold of {}: {}'.format(split_name, unk_prop_max, above_max_count))

    return src_idx_insts, tgt_idx_insts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', required=True, help='If multiple, delimit files with commas.')
    parser.add_argument('-valid_file', required=True)
    parser.add_argument('-test_file', required=True)
    parser.add_argument('-save_dir', required=True)
    parser.add_argument('-max_post_len', type=int, default=50)
    parser.add_argument('-max_disc_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=1)     # set to 1.0 to include all unique tokens
    parser.add_argument('-unk_prop_max', type=float, default=0.05)  # set to 1.0 to disregard <unk> proportions
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-use_glove_emb', action='store_true')

    opt = parser.parse_args()
    opt.max_token_post_len = opt.max_post_len + 2 # include the <s> and </s>

    ##-- training set
    print('[Info] Load training set.')
    train = []
    for train_file in opt.train_file.split(','):
        with open(train_file, 'rb') as f:
            train += pickle.load(f)
    train_src_word_insts, train_tgt_word_insts = read_instances(
        train, opt.max_post_len, opt.max_disc_len, opt.keep_case, 'train')
    # prune for mismatches and empty instances / sequences
    print('[Info] Prune empty sentences and src/tgt mismatches.')
    train_src_word_insts, train_tgt_word_insts = prune(
        train_src_word_insts, train_tgt_word_insts, 'training')
    
    ##-- validation set
    print('[Info] Load validation set.')
    with open(opt.valid_file, 'rb') as f:
        val = pickle.load(f)
    val_src_word_insts, val_tgt_word_insts = read_instances(
        val, opt.max_post_len, opt.max_disc_len, opt.keep_case, 'valid')
    # prune for mismatches and empty instances / sequences
    print('[Info] Prune empty sentences and src/tgt mismatches.')
    val_src_word_insts, val_tgt_word_insts = prune(
        val_src_word_insts, val_tgt_word_insts, 'validation')

    ##-- testing set
    print('[Info] Load testing set.')
    with open(opt.test_file, 'rb') as f:
        test = pickle.load(f)
    test_src_word_insts, test_tgt_word_insts = read_instances(
        test, opt.max_post_len, opt.max_disc_len, opt.keep_case, 'test')
    # prune for mismatches and empty instances / sequences
    print('[Info] Prune empty sentences and src/tgt mismatches.')
    test_src_word_insts, test_tgt_word_insts = prune(
        test_src_word_insts, test_tgt_word_insts, 'testing')

    ##-- build vocabulary
    if opt.vocab:
        with open(opt.vocab ,'rb') as f:
            predefined_data = pickle.load(f)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    ##-- generate glove embedding tables if using them
    if opt.use_glove_emb:
        src_word2idx, src_emb_table = create_glove_emb_table(src_word2idx, 'src')
        np.save('data/glove/src_emb_file.npy', src_emb_table)
        if opt.share_vocab:
            tgt_word2idx = src_word2idx
            tgt_emb_table = src_emb_table
        else:
            tgt_word2idx, tgt_emb_table = create_glove_emb_table(tgt_word2idx, 'tgt')
        np.save('data/glove/tgt_emb_file.npy', tgt_emb_table)

    ##-- map word to index
    print('[Info] Convert training word instances into sequences of word index.')
    train_src_insts, train_tgt_insts = convert_instance_to_idx_seq(
        zip(train_src_word_insts, train_tgt_word_insts), src_word2idx, opt.unk_prop_max, 'train')
    print('[Info] Final training set size: {}'.format(len(train_src_insts)))

    print('[Info] Convert validation word instances into sequences of word index.')
    val_src_insts, val_tgt_insts = convert_instance_to_idx_seq(
        zip(val_src_word_insts, val_tgt_word_insts), src_word2idx, opt.unk_prop_max, 'valid')
    print('[Info] Final validation set size: {}'.format(len(val_src_insts)))

    print('[Info] Convert testing word instances into sequences of word index.')
    test_src_insts, test_tgt_insts = convert_instance_to_idx_seq(
        zip(test_src_word_insts, test_tgt_word_insts), src_word2idx, opt.unk_prop_max, 'test')
    print('[Info] Final testing set size: {}'.format(len(test_src_insts)))
    
    ##-- training data
    train_data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': val_src_insts,
            'tgt': val_tgt_insts}}

    print('[Info] Dump the processed training data to pickle file', opt.save_dir + '/train.data.pt')
    torch.save(train_data, opt.save_dir + '/train.data.pt')

    ##-- testing data
    test_data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'test': {
            'src': test_src_insts,
            'tgt': test_tgt_insts}}

    print('[Info] Dump the processed testing data to pickle file', opt.save_dir + '/test.data.pt')
    torch.save(test_data, opt.save_dir + '/test.data.pt')
    
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
