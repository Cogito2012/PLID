import os
import torch
import clip
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import set_seed, get_config
from clip_modules.model_loader import load
from tqdm import tqdm
# import pickle
import pickle5 as pickle
import re



DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATHS = {
    "mit-states": os.path.join(DIR_PATH, "../data/mit-states"),
    "ut-zappos": os.path.join(DIR_PATH, "../data/ut-zappos"),
    "cgqa": os.path.join(DIR_PATH, "../data/cgqa")
}


def remove_repeating_words(sentence, maxnum_char_text):
    # Split the sentence into individual words
    words = sentence.split()
    if words[-1].endswith("."):
        words[-1] = words[-1][:-1]  # remove "." temperarily
    # Check for repeating words at the end of the sentence
    while len(words) > 1 and words[-1] == words[-2]:
        # If the last two words are the same, remove the last word
        words.pop()
    # Join the remaining words back into a sentence
    modified_sentence = ' '.join(words)
    if len(modified_sentence) > maxnum_char_text:
        modified_sentence = modified_sentence[:maxnum_char_text]
    modified_sentence +=  '.'
    return modified_sentence


def is_mostly_letters(string):
    # Remove all whitespace and punctuation from the string
    clean_string = re.sub(r'[^\w\s]', '', string)
    
    # Count the number of letters and non-letter characters
    letter_count = sum(1 for c in clean_string if c.isalpha())
    non_letter_count = len(clean_string) - letter_count
    
    # Check if more than half of the characters are letters
    if letter_count > non_letter_count:
        return True
    
    return False


def compute_db_features(clip_model, data_root, num_texts, config, device, maxnum_words_text=20, maxnum_char_word=30, maxnum_char_text=128):
    db_filepath = os.path.join(data_root, config.text_db)
    with open(db_filepath, 'rb') as f:
        data = pickle.load(f)
        refined_data = dict()
        for k, v in data.items():  
            # select the first 64 texts as a batch
            text_batch = []
            for i in range(num_texts):
                words = v[i].split()[:maxnum_words_text]  # each sentence is no longer than 20 words
                if len(words) == 0:
                    sentence = 'a photo of {} {}.'.format(k[0], k[1])
                else:
                    sentence = ' '.join(word[:maxnum_char_word] for word in words)  # each word is no longer than 30 letters
                    sentence = remove_repeating_words(sentence, maxnum_char_text)  # remove repeating words and add .
                    if not is_mostly_letters(sentence) or '#####' in sentence:
                        sentence = 'a photo of {} {}.'.format(k[0], k[1])
                text_batch.append(sentence)
            refined_data[k] = text_batch
    # compute CLIP features
    from clip_modules.text_encoder import CustomTextEncoder
    text_encoder = CustomTextEncoder(clip_model, clip_model.dtype, device)
    text_features = {}
    with torch.no_grad():
        for pair, text_batch in tqdm(refined_data.items(), total=len(refined_data), ncols=0):
            token_ids = clip.tokenize(text_batch, context_length=config.context_length).to(device)  # (N, ctxlen)
            text_features[pair], _ = text_encoder(token_ids, None, enable_pos_emb=True)
    return text_features


if __name__ == "__main__":
    """ CUDA_VISIBLE_DEVICES=1 python exp/compute_db_features.py --text_db t5_sentences200_all_pairs_open.pkl --clip_model ViT-L/14 --context_length 77 --dataset mit-states  
    """

    # get input configurations
    config = get_config()

    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_path = DATASET_PATHS[config.dataset]
    
    clip_model, preprocess = load(config.clip_model, device=device)

    feat_cache = os.path.join(dataset_path, config.text_db[:-4] + '_feat.pkl')  # t5_sentences200_all_pairs.pkl
    num_texts = getattr(config, 'num_texts', 32)
    if not os.path.exists(feat_cache):
        print("Computing the features of text database...")
        text_features = compute_db_features(clip_model, dataset_path, num_texts, config, device)
        with open(feat_cache, "wb") as f:
            pickle.dump(text_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("The pre-computed features of text database already exist!{}".format(feat_cache))