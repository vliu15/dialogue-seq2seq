#!/usr/local/bin/python
from grab_data.discussion import Dataset, results_root_dir, data_root_dir
from nltk import word_tokenize
import unicodedata
import random
import pickle

def load_data():
    data = []
    dataset = Dataset(name='fourforums')
    for discussion in dataset.get_discussions():
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
    return data

def main():
    data = load_data()
    random.shuffle(data)

    total = len(data)

    train = data[total//10:]
    val = data[total//20:total//10]
    test = data[:total//20]
    print("[Info] Data split into {}, {}, {} samples for training, validation, and testing.".format(len(train), len(val), len(test)))

    print('[Info] Saving validation set...')
    with open('../val.pkl', 'wb') as f:
        pickle.dump(val, f)
    print('[Info] Saving testing set...')
    with open('../test.pkl', 'wb') as f:
        pickle.dump(test, f)
    print('[Info] Saving training set...')
    with open('../train.pkl', 'wb') as f:
        pickle.dump(train, f)

if __name__ == "__main__":
    main()
