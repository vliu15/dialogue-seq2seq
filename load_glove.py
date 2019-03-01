import numpy as np
import pickle
import argparse
from tqdm import tqdm

from transformer import Constants

def load_glove(glove_path, glove_size):
    print("[Info] Load GloVe model.")
    with open(glove_path,'r', encoding="utf-8") as f:
        word2idx = {
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS,
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK}
        idx2emb = {
            Constants.BOS: np.zeros(shape=(glove_size)),
            Constants.EOS: np.zeros(shape=(glove_size)),
            Constants.PAD: np.zeros(shape=(glove_size)),
            Constants.UNK: np.zeros(shape=(glove_size)),
        }

        for idx, line in tqdm(enumerate(f, 4)):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            word2idx[word] = idx
            idx2emb[idx] = embedding

    #- Create embedding table
    emb_table = np.zeros(shape=(len(word2idx), glove_size))
    for i in range(len(word2idx)):
        emb_table[i] = idx2emb[i]
    print(emb_table.shape)

    #- Create word2idx mapping
    word2idx = {
        'dict': {
            'src': word2idx,
            'tgt': word2idx
        }}
    
    #- Save vocabulary and embeddings
    with open('data/glove/word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    np.save('data/glove/emb_table.npy', emb_table)

    return word2idx, emb_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-glove_file', default='data/glove/glove.6B.300d.txt', help='Path to GloVe model file')
    parser.add_argument('-glove_size', default=300, help='GloVe word embedding size')
    args = parser.parse_args()

    _, _ = load_glove(args.glove_file, args.glove_size)

if __name__ == "__main__":
    main()
