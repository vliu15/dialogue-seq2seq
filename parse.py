from data.iac.code.grab_data.discussion import Dataset, results_root_dir, data_root_dir
import unicodedata
from nltk import word_tokenize
import random
from transformer.Constants import Constants
import argparse


def load_data():
    i = 0
    data = []
    dataset = Dataset(name='fourforums', annotation_list=['topic','mechanical_turk'])
    for discussion in dataset.get_discussions(annotation_label='mechanical_turk'):
        thread = {}
        posts = []
        # discussion.posts is a dict of posts
        for post in discussion.get_posts():
            text_without_quotes = post.delete_ranges('quotes')
            ascii_text = text_without_quotes.encode('ascii', 'ignore').replace('\n', '') 
            posts.append(word_tokenize(ascii_text))
        thread["src"] = posts[:-1]
        thread["tgt"] = posts[1:]
        data.append(thread)
        print(i)
    return data

def process_sequence(seq, max_post_len, max_disc_len, keep_case):
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


def read_instances(inst_list, max_post_len, max_disc_len, keep_case, split_name):
    src_insts = []
    tgt_insts = []
    trimmed_disc_count_src = 0
    trimmed_post_count_src = 0
    trimmed_disc_count_tgt = 0
    trimmed_post_count_tgt = 0
    for disc in inst_list:
        src_inst, tdcs, tpcs = process_sequence(disc["src"], max_post_len, max_disc_len, keep_case)
        tgt_inst, tdct, tpct = process_sequence(disc["tgt"], max_post_len, max_disc_len, keep_case)

        src_insts.append(src_inst)
        tgt_insts.append(tgt_inst)

        trimmed_disc_count_src += tdcs
        trimmed_post_count_src += tpcs
        trimmed_disc_count_tgt += tdct
        trimmed_post_count_tgt += tpct


    print('[Info] Get {} instances from {}'.format(len(src_insts), split_name + '-src'))
    print('[Info] Get {} instances from {}'.format(len(tgt_insts), split_name + '-tgt'))

    print('[Info] {}: {} instances are trimmed to the max discussion length {}'
        .format(split_name + '-src', trimmed_disc_count_src, max_disc_len))
    print('[Info] {}: {} subinstances are trimmed to the max post length {}'
        .format(split_name + '-src', trimmed_post_count_src, max_post_len))
    print('[Info] {}: {} instances are trimmed to the max discussion length {}'
        .format(split_name + '-tgt', trimmed_disc_count_tgt, max_disc_len))
    print('[Info] {}: {} subinstances are trimmed to the max post length {}'
        .format(split_name + '-tgt', trimmed_post_count_tgt, max_post_len))

    return src_insts, tgt_insts

def main():
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_post_len', type=int, default=50)
    parser.add_argument('-max_disc_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_post_len = opt.max_post_len + 2 # include the <s> and </s>

    data = load_data()
    random.shuffle(data)
    test = data[:590]
    val = data[590:1180]
    train = data[1180:]

    train_src_word_insts, train_tgt_word_insts = read_instances(
        train, opt.max_post_len, opt.max_disc_len, opt.keep_case, "train")
    val_src_word_insts, val_tgt_word_insts = read_instances(
        val, opt.max_post_len, opt.max_disc_len, opt.keep_case, "val")
    test_src_word_insts, test_tgt_word_insts = read_instances(
        val, opt.max_post_len, opt.max_disc_len, opt.keep_case, "test")


if __name__ == "__main__":
    main()
