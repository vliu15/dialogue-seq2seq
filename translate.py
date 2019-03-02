''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import pickle

from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-test_file', required=True, help='Test pickle file for validation')
    parser.add_argument('-output', default='outputs.txt', help='Path to output the predictions (each line will be the decoded sequence')
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-n_best', type=int, default=1, help='If verbose is set, will output the n_best decoded sentences')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    test_data = torch.load(opt.test_file)

    test_src_insts = test_data['test']['src']
    test_tgt_insts = test_data['test']['tgt']

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=test_data['dict']['src'],
            tgt_word2idx=test_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        drop_last=True,
        collate_fn=collate_fn)

    translator = Translator(opt)

    print('[Info] Evaluate on test set.')
    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test / Discussions)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch) # structure: List[batch, seq, pos]
            for disc in all_hyp:
                f.write('[')
                for post in disc:
                    post = post[0]
                    pred_post = ' '.join([test_loader.dataset.tgt_idx2word[word] for word in post])
                    f.write('\t' + pred_post + '\n')
                f.write(']\n')
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
