#!/usr/local/bin/python
from grab_data.discussion import Dataset, results_root_dir, data_root_dir
import unicodedata
import random
import pickle

def load_data():
    data = []
    dataset = Dataset(name='fourforums')
    seen = set([])
    dup = 0
    for discussion in dataset.get_discussions():
        thread = {}
        posts = []
        # discussion.posts is a dict of posts
        for post in discussion.get_posts():
            text = post.delete_ranges('quotes')
            text = text.encode('ascii', 'ignore').replace('\n', '')
            if text in seen:
                dup += 1
                continue
            else:
                seen.add(text)
            posts.append(text)

        thread["src"] = posts[:-1]
        thread["tgt"] = posts[1:]
        data.append(thread)

    print('[Info] Number of duplicate posts: {}'.format(dup))

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
    shards = len(train)//4
    with open('../train.1.pkl', 'wb') as f:
        pickle.dump(train[:shards], f)
    with open('../train.2.pkl', 'wb') as f:
        pickle.dump(train[shards:2*shards], f)
    with open('../train.3.pkl', 'wb') as f:
        pickle.dump(train[2*shards:3*shards], f)
    with open('../train.4.pkl', 'wb') as f:
        pickle.dump(train[3*shards:], f)

if __name__ == "__main__":
    main()
