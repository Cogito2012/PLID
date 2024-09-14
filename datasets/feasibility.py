import argparse
import os
from itertools import product

import clip
import torch
import torch.nn.functional as F
import sys
root_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, root_path)

from datasets.composition_dataset import CompositionDataset, _preprocess_utzappos
from datasets.read_datasets import DATASET_PATHS
from clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to(device)


def load_glove_embeddings(vocab):
    '''
    Inputs
        emb_file: Text file with word embedding pairs e.g. Glove, Processed in lower case.
        vocab: List of words
    Returns
        Embedding Matrix
    '''
    vocab = [v.lower() for v in vocab]
    emb_file = os.path.join(DIR_PATH, 'data/feasibility_glove/glove.6B.300d.txt')
    model = {}  # populating a dictionary of word and embeddings
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')  # Word-embedding
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        model[line[0]] = wvec

    # Adding some vectors for UT Zappos
    custom_map = {
        'faux.fur': 'faux_fur',
        'faux.leather': 'faux_leather',
        'full.grain.leather': 'full_grain_leather',
        'hair.calf': 'hair_calf_leather',
        'patent.leather': 'patent_leather',
        'boots.ankle': 'ankle_boots',
        'boots.knee.high': 'knee_high_boots',
        'boots.mid-calf': 'midcalf_boots',
        'shoes.boat.shoes': 'boat_shoes',
        'shoes.clogs.and.mules': 'clogs_shoes',
        'shoes.flats': 'flats_shoes',
        'shoes.heels': 'heels',
        'shoes.loafers': 'loafers',
        'shoes.oxfords': 'oxford_shoes',
        'shoes.sneakers.and.athletic.shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower',
        'nubuck' : 'grainy_leather',
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k:
            ks = k.split('_')
            emb = torch.stack([model[it] for it in ks]).mean(dim=0)
        elif ' ' in k:
            emb = torch.stack([model[it] for it in k.split(' ')]).mean(dim=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.stack(embeds)
    print('Glove Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds


def clip_embeddings(model, words_list):
    words_list = [word.replace(".", " ").lower() for word in words_list]
    prompts = [f"a photo of {word}" for word in words_list]

    tokenized_prompts = clip.tokenize(prompts)
    with torch.no_grad():
        _text_features = model.text_encoder(tokenized_prompts, enable_pos_emb=True)
        text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        return text_features

def get_pair_scores_objs(attr, obj, all_objs, attrs_by_obj_train, obj_embedding_sim):
    score = -1.
    for o in all_objs:
        if o!=obj and attr in attrs_by_obj_train[o]:
            temp_score = obj_embedding_sim[(obj,o)]
            if temp_score>score:
                score=temp_score
    return score

def get_pair_scores_attrs(attr, obj, all_attrs, obj_by_attrs_train, attr_embedding_sim):
    score = -1.
    for a in all_attrs:
        if a != attr and obj in obj_by_attrs_train[a]:
            temp_score = attr_embedding_sim[(attr, a)]
            if temp_score > score:
                score = temp_score
    return score


def compute_feasibility(dataset, attr_embeddings, obj_embeddings):
    objs = dataset.objs
    attrs = dataset.attrs

    obj_embedding_sim = compute_cosine_similarity(objs, obj_embeddings,
                                                        return_dict=True)
    attr_embedding_sim = compute_cosine_similarity(attrs, attr_embeddings,
                                                        return_dict=True)

    print('computing the feasibilty score')
    feasibility_scores = dataset.seen_mask.clone().float()
    for a in attrs:
        for o in objs:
            if (a, o) not in dataset.train_pairs:
                idx = dataset.pair2idx[(a, o)]
                score_obj = get_pair_scores_objs(
                    a,
                    o,
                    dataset.objs,
                    dataset.attrs_by_obj_train,
                    obj_embedding_sim
                )
                score_attr = get_pair_scores_attrs(
                    a,
                    o,
                    dataset.attrs,
                    dataset.obj_by_attrs_train,
                    attr_embedding_sim
                )
                score = (score_obj + score_attr) / 2
                feasibility_scores[idx] = score

    # feasibility_scores = feasibility_scores

    return feasibility_scores * (1 - dataset.seen_mask.float())


def preprocess_pairs(all_objs, all_attrs):
    # cleaning the classes and the attributes
    if config.dataset == 'ut-zappos':
        # cleaning the classes and the attributes
        objects, attributes = _preprocess_utzappos(all_objs, all_attrs)
    else:
        objects = [cla.replace(".", " ").lower() for cla in all_objs]
        attributes = [attr.replace(".", " ").lower() for attr in all_attrs]
    return objects, attributes


def compute_clip_embeddings(dataset, feasible_pairs, ctx_len=16):
    # load the pretrained CLIP model
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=ctx_len
    )
    # preprocess pairs of words
    all_attrs = [attr for attr, _ in feasible_pairs]
    all_objs = [obj for _, obj in feasible_pairs]
    all_attrs, all_objs = preprocess_pairs(all_attrs, all_objs)
    processed_pairs = list(zip(all_attrs, all_objs))
    num_pairs = len(processed_pairs)

    attrs, objs = preprocess_pairs(dataset.attrs, dataset.objs)
    
    # tokenize the prompted text for each pair
    tokenized_text = torch.cat([
        clip.tokenize('a photo of {} {}.'.format(attr, obj), context_length=ctx_len)
        for (attr, obj) in processed_pairs
    ])  # (N, 16)

    # get the text embedding for each pair
    with torch.no_grad():
        all_token_embeddings = clip_model.token_embedding(tokenized_text.to(device)) # (N, 16, D)
        text_embedding = torch.zeros(
            (num_pairs, clip_model.token_embedding.weight.size(-1)),
        ).to(device)
        for idx, rep in enumerate(all_token_embeddings):
            eos_idx = tokenized_text[idx].argmax()
            text_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    # group text embeddings by attr
    attr_embeddings = torch.zeros((len(attrs), text_embedding.size(-1))).to(device)
    for n, attr in enumerate(attrs):
        attr_embeddings[n] = torch.cat([text_embedding[[i]] for i, p in enumerate(processed_pairs) if p[0] == attr], dim=0).mean(dim=0)
    
    # group text embeddings by obj
    obj_embeddings = torch.zeros((len(objs), text_embedding.size(-1))).to(device)
    for n, obj in enumerate(objs):
        obj_embeddings[n] = torch.cat([text_embedding[[i]] for i, p in enumerate(processed_pairs) if p[1] == obj], dim=0).mean(dim=0)
    
    return attr_embeddings, obj_embeddings



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default=None,
    )
    config = parser.parse_args()

    dataset_path = DATASET_PATHS[config.dataset]
    test_dataset =  CompositionDataset(dataset_path,
                                    phase='test',
                                    split='compositional-split-natural',
                                    open_world=True)

    if config.clip_model is None:
        # use Glove model to compute feasibility
        attr_embeddings = load_glove_embeddings(test_dataset.attrs).to(device)
        obj_embeddings = load_glove_embeddings(test_dataset.objs).to(device)
        save_dir = 'data/feasibility_glove'
    else:
        closed_dataset = CompositionDataset(dataset_path,
                                    phase='test',
                                    split='compositional-split-natural',
                                    open_world=False)
        # use CLIP text encoder to compute feasibility
        attr_embeddings, obj_embeddings = compute_clip_embeddings(test_dataset, closed_dataset.pairs)
        save_dir = 'data/feasibility_clip'

    feasibility = compute_feasibility(test_dataset, attr_embeddings, obj_embeddings)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(DIR_PATH, f'{save_dir}/feasibility_{config.dataset}.pt')
    torch.save({
        'feasibility': feasibility,
    }, save_path)

    print('done!')
