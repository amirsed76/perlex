import pickle

import torch


def save_obj(obj, name, dir='./'):
    with open(dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, dir='./'):
    with open(dir + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def e_masks(should_mask, config):
    masked = []
    for start, end in should_mask:
        if config['model_name'] == 'xlm-roberta-base':
            mask = torch.zeros(config['max_length'])
            mask[start:end] = 1
        if config['model_name'] == 'xlm-roberta-large':
            mask = torch.zeros(config['max_length'])
            mask[start:end + 1] = 1
        if config['model_name'] == 'bert-base-multilingual-cased':
            mask = torch.zeros(config['max_length'])
            mask[start:end] = 1
        if config['model_name'] == 'HooshvareLab/bert-fa-zwnj-base':
            mask = torch.zeros(config['max_length'])
            mask[start:end + 1] = 1
        masked.append(mask.tolist())
    return masked
