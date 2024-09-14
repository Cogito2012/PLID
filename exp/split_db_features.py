import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from datasets.composition_dataset import pairwise_class_names
from tqdm import tqdm


if __name__ == '__main__':
    """ CUDA_VISIBLE_DEVICES=1 python exp/split_db_features.py cgqa
    """

    dataset = sys.argv[1]
    assert dataset in ['mit-states', 'ut-zappos', 'cgqa'], "Unsupported dataset: {}".format(dataset)
    data_root = os.path.join('data', dataset)

    data_info = pairwise_class_names(data_root, 'compositional-split-natural')
    all_pairs = data_info['all_pairs']

    db_file = os.path.join(data_root, 'opt_sentences64_all_pairs_open_feat.pkl')
    with open(db_file, "rb") as f:
        text_features = pickle.load(f)

    all_pairs_feat = dict()
    for i, pair in tqdm(enumerate(all_pairs), ncols=0, desc='processing'):
        if pair in text_features:
            all_pairs_feat[pair] = text_features[pair]
    
    save_file = os.path.join(data_root, 'opt_sentences64_all_pairs_closed_feat.pkl')
    with open(save_file, "wb") as f:
        pickle.dump(all_pairs_feat, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Done!")