''' Translate input text with trained model '''
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import pickle
from dataset import collate_fn, TranslationDataset
from seq2seq.Translator import Translator


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .chkpt file')
    parser.add_argument('-test_file', required=True, help='Test pickle file for validation')
    parser.add_argument('-output', default='outputs.txt', help='Path to output the predictions (each line will be the decoded sequence')
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-n_best', type=int, default=1, help='If verbose is set, will output the n_best decoded sentences')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #- Prepare Translator
    translator = Translator(opt)
    print('[Info] Model opts: {}'.format(translator.model_opt))

    #- Prepare DataLoader
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

    print('[Info] Evaluate on test set.')
    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Testing)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch) # structure: List[batch, seq, pos]
            for inst in all_hyp:
                f.write('[')
                for seq in inst:
                    seq = seq[0]
                    pred_seq = ' '.join([test_loader.dataset.tgt_idx2word[word] for word in seq])
                    f.write('\t' + pred_seq + '\n')
                f.write(']\n')
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
