import os, sys
import torch
import pickle
import numpy as np
import random
import time
import re
from threading import Thread
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from datasets.composition_dataset import pairwise_class_names

from transformers import AutoTokenizer
modelname = 't5' if len(sys.argv) <= 2 else sys.argv[2]
assert modelname in ['t5', 'opt', 'mistral7b', 'gpt3', 'gpt3.5', 'gptj6b']
if modelname == 't5':
    from transformers import AutoModelForSeq2SeqLM
if modelname == 'opt':
    from transformers import OPTForCausalLM
if modelname in ['gpt3', 'gpt3.5']:
    import openai
    openai.api_key = 'YOUR_API_KEY'
    from openai import OpenAI
    client = OpenAI(api_key='YOUR_API_KEY')
if modelname == 'gptj6b':
    from transformers import GPTJForCausalLM
if modelname == 'mistral7b':  # need at least Python>=3.8 and transformers >= 4.37.2
    from transformers import AutoModelForCausalLM
import random
random.seed(42)


def load_k2t_model(modelname, use_cuda=True, device=None):
    if device is None:
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if modelname == 't5':
        model_path = './pretrained/t5-base-finetuned-common_gen'
        k2t_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif modelname == 'opt':
        model_path = './pretrained/opt-1.3b'
        k2t_model = OPTForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif modelname == 'gptj6b':
        model_path = './pretrained/gpt-j-6b'
        k2t_model = GPTJForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif modelname == 'mistral7b':
        model_path = './pretrained/Mistral-7B-Instruct-v0.2'
        k2t_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        k2t_model, tokenizer = None, None
    return k2t_model, tokenizer


def remove_repeating_words(sentence):
    # Split the sentence into individual words
    words = sentence.split()
    if words[-1].endswith("."):
        words[-1] = words[-1][:-1]  # remove "." temperarily
    # Check for repeating words at the end of the sentence
    while len(words) > 1 and words[-1] == words[-2]:
        # If the last two words are the same, remove the last word
        words.pop()
    # Join the remaining words back into a sentence
    modified_sentence = ' '.join(words) + '.'
    return modified_sentence


def generate_t5_sentences(key, k2t_model, tokenizer, num_gen=10):
    # generate token ids
    input_ids = tokenizer("{} </s>".format(key), \
        max_length=1024, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt').input_ids
    results = []
    while len(results) < num_gen:
        # conditional generation
        outputs = k2t_model.generate(input_ids.to("cuda"), \
            num_return_sequences=num_gen-len(results), do_sample=True, max_new_tokens=1024, temperature=0.7)
        # decode token ids to text
        for out in outputs:
            text = tokenizer.decode(out)  # tokens to texts
            text = re.sub("<pad>|</s>", "", text)  # remove padding words
            if '<unk>' in text:
                continue  # ignore text containing '<unk>'
            text = text.strip().lower()   # remove white spaces and lower letters
            text = remove_repeating_words(text)  # remove repeating words and add .
            if text not in results:
                results.append(text)
    return results


def generate_opt_sentences(key, k2t_model, tokenizer, num_gen=10):
    training_cases = """Keywords: sliced, potato, picture
Output: The picture features a beautifully arranged plate of thinly sliced potatoes.
###
Keywords: red, apple, photo
Output: In the photo, a bright red apple is the central focus, captured in stunning detail.
###
Keywords: leather, shoes, image
Output: The image showcases a sleek pair of leather shoes, meticulously crafted and designed to impress.
###
Keywords: broken, car, photo
Output: The photo captures the aftermath of a car accident, with a broken vehicle lying on the road. 
###"""
    keywords_string = ", ".join(key.split(' ')) + ', {}'.format(random.choice(['photo', 'image', 'picture']))
    prompt = training_cases + "\nKeywords: "+ keywords_string + "\nOutput: "
    input_ids = tokenizer(prompt, \
        add_special_tokens=False, truncation=True, padding=True, return_tensors='pt').input_ids
    results = []
    while len(results) < num_gen:
        # conditional generation
        outputs = k2t_model.generate(input_ids.to("cuda"), \
            num_return_sequences=num_gen, do_sample=True, max_new_tokens=32, no_repeat_ngram_size=2)
        # decode token ids to text
        for out in outputs:
            text = tokenizer.decode(out)  # tokens to texts
            text = re.sub("<pad>|</s>", "", text)  # remove padding words
            text = text[len(prompt):]
            if '<unk>' in text:
                continue  # ignore text containing '<unk>'
            text = text.strip().lower()   # remove white spaces and lower letters
            if len(text) < 5:  # a sentence should never be less than 5 chars
                continue
            text = remove_repeating_words(text)  # remove repeating words and add .
            if text not in results:
                results.append(text)
    return results[:num_gen]


def generate_mistral_sentences(key, k2t_model, tokenizer, num_gen=10):
    prompts = "Please use one short sentence (less than 20 words) to describe the visual appearance of a photo of {}. The sentences should be independently generated. Return {} sentences in a list.".format(key, num_gen)
    messages = [# {"role": "system", "content": "You are a useful assistant."}, 
                {"role": "user", "content": prompts}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    input_ids = input_ids.to(k2t_model.device)
    results = []
    while len(results) < num_gen:
        # generate
        generated_ids = k2t_model.generate(input_ids, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(generated_ids)
        ans = decoded[0].split('[/INST]')[-1].strip().strip('</s>')
        outputs = ans.split("\n")
        for out in outputs:
            text = re.sub(r'\d+. ', '', out)
            if text not in results:
                results.append(text)
    return results[:num_gen]



def generate_sentence_prompt(attr: str, obj: str, num_gen: int):
    return f"""Q: How do you describe a photo of red apple? 
Please give at least 10 sentences with different sentence structures. Each sentence contains both "red" and "apple".
A: Here is a list of sentences that describe a photo of red apple:
- A red apple sits in the photo.
- The photo shows a red apple.
- A red apple is pictured.
- A red apple is the focus of the photo.
- The photo captures a red apple.
- A red apple is the subject of the photo.
- A red apple is the star of the photo.
- The photo displays a red apple.
- A red apple is pictured in the photo.
- The photo features a red apple.
- A red apple sits in the frame of the photo.
- The photo captures a single red apple.
Q: How do you describe a photo of {attr} {obj}? 
Please give at least {num_gen} sentences with different sentence structures. Each sentence contains both "{attr}" and "{obj}".
A: Here is a list of sentences that describe a photo of {attr} {obj}:
-
"""

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_response(description, minlen):
    result = []
    for sentence in description.split('\n'):
        sentence = sentence.strip()  # remove white spaces at the begining and end of a string
        if sentence == '' or len(sentence) <= minlen:
            continue
        if sentence.startswith('- '):
            sentence = sentence[2:]  # e.g. "- A vivid red apple hanging on a tree branch."
        if not sentence[-1].endswith('.'):
            sentence += '.'  # a complete sentence ends with '.'
        result.append(sentence)
    return result


def parse_gptj_response(description, attr, obj):
    minlen = len(attr)+len(obj)+1
    result = []
    for sentence in description.split("\n"):
        sentence = sentence.strip()  # remove white spaces at the begining and end of a string
        if sentence == '' or len(sentence) <= minlen or (attr not in sentence) or (obj not in sentence):
            continue
        if sentence.startswith('- '):
            sentence = sentence[2:]  # e.g. "- A vivid red apple hanging on a tree branch."
        if not sentence[-1].endswith('.'):
            sentence += '.'  # a complete sentence ends with '.'
        result.append(sentence)
    return result


def generate_gptj_sentences(attr, obj, model, tokenizer, num_gen=200, device=torch.device("cuda"), print_info=False):

    result_full = []
    i = 1
    while len(result_full) < num_gen:
        set_seed(i * 42)
        # get the text prompt
        prompt = generate_sentence_prompt(attr, obj, num_gen=i*(num_gen-len(result_full)))

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(input_ids.to(device), do_sample=True, temperature=0.5, max_length=1024, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        result = parse_gptj_response(gen_text, attr, obj)
        # add results
        for res in result:
            if res not in result_full:
                result_full.append(res)
        i += 1
    result_full = result_full[:num_gen]

    if print_info:
        print("==> Totally {} sentences generated:".format(len(result)))
        for res in result_full:
            print(res)
    
    return result_full



def generate_gpt3_sentences(attr, obj, num_gen=200, print_info=False):
    result_full = []
    i = 1
    while len(result_full) < num_gen:
        set_seed(i * 42)
        # get the text prompt
        prompt = generate_sentence_prompt(attr, obj, num_gen=i*(num_gen-len(result_full)))
        # generate GPT-3 responses
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=1024, n=5)
        result = parse_response(response['choices'][0]['text'], minlen=len(attr)+len(obj)+1)
        # add results
        for res in result:
            if res not in result_full:
                result_full.append(res)
        i += 1
    result_full = result_full[:num_gen]
    
    if print_info:
        print("==> Totally {} sentences generated:".format(len(result)))
        for res in result_full:
            print(res)

    return result_full

def generate_gpt35_sentences(attr, obj, num_gen=64):

    prompt = f"""Generate {num_gen + 10} short sentences that each sentence visually describes an image/photo/picture of '{attr} {obj}'. 
For example, given a photo of `red apple`, output sentences like:- A red apple sits in the photo.
- The photo shows a red apple.
- A red apple is pictured.
- A red apple is the focus of the photo.
- The photo captures a red apple.
- A red apple is the subject of the photo.
- A red apple is the star of the photo.
- The photo displays a red apple.
- A red apple is pictured in the photo.
- The photo features a red apple.
- A red apple sits in the frame of the photo.
- The photo captures a single red apple."""
    message = [
        {"role": "system", "content": "You are a useful image content captioner."},
        {"role": "user", "content": prompt}
    ]
    results = []
    while len(results) < num_gen:
        # generate
        response = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=message, max_tokens=800)  # Adjust max_tokens as needed
        res = response.choices[0].message.content.strip().split('\n')
        results.extend(res)
    return results[:num_gen]


def single_process(pairs, result, k2t_model=None, tokenizer=None, num_gen=64):
    for i, (attr, obj) in enumerate(pairs):
        key = '{} {}'.format(attr, obj)
        t = time.time()
        if modelname in ['t5', 'flant5']:
            sentences = generate_t5_sentences(key, k2t_model, tokenizer, num_gen=num_gen)
        if modelname in 'opt':
            sentences = generate_opt_sentences(key, k2t_model, tokenizer, num_gen=num_gen)
        if modelname == 'mistral7b':
            sentences = generate_mistral_sentences(key, k2t_model, tokenizer, num_gen=num_gen)
        elif modelname == 'gptj6b':
            sentences = generate_gptj_sentences(attr, obj, k2t_model, tokenizer, num_gen=num_gen, device=torch.device('cuda'))
        elif modelname == 'gpt3':
            sentences = generate_gpt3_sentences(attr, obj, num_gen=num_gen)
        elif modelname == 'gpt3.5':
            sentences = generate_gpt35_sentences(attr, obj, num_gen=num_gen)
        result[(attr, obj)] = sentences
        seconds = time.time() - t
        print("Model: {}, process pair: {}, progress: {}/{}, time: {:.3f}s.".format(modelname, key, i+1, len(pairs), seconds))


def main():
    dataset = sys.argv[1]
    assert dataset in ['mit-states', 'ut-zappos', 'cgqa'], "Unsupported dataset: {}".format(dataset)

    data_root = os.path.join('data', dataset)
    num_gen = 64  # for each combinational class, we generate 200 sentences
    open_world = True

    filename = f'{modelname}_sentences{num_gen}_all_pairs'
    if open_world: filename += '_open'
    save_file = os.path.join(data_root, filename + '.pkl')
    if not os.path.exists(save_file):
        # load the T5 langauge model
        k2t_model, tokenizer = load_k2t_model(modelname, use_cuda=True)
        
        result = {}
        data_info = pairwise_class_names(data_root, 'compositional-split-natural')
        pairs = data_info['all_pairs_open'] if open_world else data_info['all_pairs']

        num_procs = 8
        pairs_split = np.array_split(pairs, num_procs)
        result = {p: None for p in pairs}
        threads = [None] * num_procs
        for i in range(num_procs):
            threads[i] = Thread(target=single_process, args=(pairs_split[i], result, k2t_model, tokenizer, num_gen))
        
        for i in range(num_procs):
            threads[i].start()

        for i in range(num_procs):
            threads[i].join()
        
        with open(save_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Result file already exist!\n{}".format(save_file))


def main_worker(pairs, k2t_model, tokenizer, num_gen, result):
    for i, (attr, obj) in enumerate(pairs):
        key = '{} {}'.format(attr, obj)
        t = time.time()
        sentences = generate_mistral_sentences(key, k2t_model, tokenizer, num_gen=num_gen)
        result[(attr, obj)] = sentences
        seconds = time.time() - t
        print("Model: {}, process pair: {}, progress: {}/{}, time: {:.3f}s.".format(modelname, key, i+1, len(pairs), seconds))


def main_multigpu():
    dataset = sys.argv[1]
    assert dataset in ['mit-states', 'ut-zappos', 'cgqa'], "Unsupported dataset: {}".format(dataset)

    available_gpus = {i: torch.device("cuda:{}".format(i)) for i in range(torch.cuda.device_count())}
    num_gpus = min(len(available_gpus), 8)

    data_root = os.path.join('data', dataset)
    num_gen = 32  # for each combinational class, we generate 200 sentences
    open_world = True

    filename = f'{modelname}_sentences{num_gen}_all_pairs'
    if open_world: filename += '_open'
    save_file = os.path.join(data_root, filename + '.pkl')
    if not os.path.exists(save_file):
        # # load the T5 langauge model
        # k2t_model, tokenizer = load_k2t_model(modelname, use_cuda=False)
        
        data_info = pairwise_class_names(data_root, 'compositional-split-natural')
        pairs = data_info['all_pairs_open'] if open_world else data_info['all_pairs']
        pairs_split = np.array_split(pairs, num_gpus)
        result = {p: None for p in pairs}

        threads = [None] * num_gpus
        for i in range(num_gpus):
            k2t_model, tokenizer = load_k2t_model(modelname, use_cuda=True, device=available_gpus[i])
            threads[i] = Thread(target=main_worker, args=(pairs_split[i], k2t_model, tokenizer, num_gen, result))
        
        for i in range(num_gpus):
            threads[i].start()

        for i in range(num_gpus):
            threads[i].join()

        with open(save_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Result file already exist!\n{}".format(save_file))


if __name__ == '__main__':

    """CUDA_VISIBLE_DEVICES=0 python -u exp/text_augment.py cgqa
    """

    # # demo
    # k2t_model = load_t5_model('./pretrained/t5-base-finetuned-common_gen', use_cuda=True)
    # generate_sentences(k2t_model, key='red apple', strict=True, num_gen=10, print_info=True)
    
    main()

    # main_multigpu()

    # result = generate_gpt3_sentences(attr='red', obj='apple', num_gen=200, print_info=True)
