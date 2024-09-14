import copy
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from pprint import pformat

from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from utils import get_config, Evaluator

cudnn.benchmark = True


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def load_trained_model(config, val_dataset, device):
    if config.experiment_name == 'clip':
        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.context_length)
        model = CLIPInterface(
            clip_model,
            config,
            token_ids=None,
            device=device,
            enable_pos_emb=True)
    elif config.experiment_name in ['condcsp', 'condcsp_proda']:
        model, optimizer = get_model(val_dataset, config, device, is_training=False)
        model.set_soft_embeddings(config.soft_embeddings)
    else:
        model, optimizer = get_model(val_dataset, config, device, is_training=False)
        if getattr(config, 'soft_embeddings', None) is not None:
            model.set_soft_embeddings(config.soft_embeddings)
        elif getattr(config, 'ckpt_file', None) is not None:
            checkpoint = torch.load(config.ckpt_file, map_location=device)
            model.load_state_dict(checkpoint['model'])
    
    if hasattr(config, 's'):
        model.clip_model.logit_scale.data = torch.tensor(config.s).log()

    return model


def get_text_representations(model, dataset, config, norm=True, get_feat=False):
    # get text representations
    text_rep = None  # conditioned on images
    # these three methods need to divide the attr-obj pairs into groups
    if config.experiment_name in ['gencsp']:
        model.set_pairs_group(dataset)
    # these three methods rely on images to compute text representations
    if not config.experiment_name in ['gencsp']:
        text_rep = model.compute_text_representations(dataset, norm=True)
    if config.experiment_name == 'gencsp':
        text_rep = model.compute_text_representations(dataset, norm=False, get_feat=get_feat)
    return text_rep


def load_feasibilities(config):
    feasible_type = getattr(config, 'feasibility_model', 'glove')
    feasibility_path = os.path.join(
        DIR_PATH, f'data/feasibility_{feasible_type}/feasibility_{config.dataset}.pt')
    unseen_scores = torch.load(
        feasibility_path,
        map_location='cpu')['feasibility']
    return unseen_scores


def search_feasibility_threshold(model, val_text_rep, val_dataset, dataloader, unseen_scores, evaluator, print_info=True):
    # decide the searching space by unseen_scores
    seen_mask = val_dataset.seen_mask.to('cpu')
    min_feasibility = (unseen_scores + seen_mask * 10.).min()
    max_feasibility = (unseen_scores - seen_mask * 10.).max()
    thresholds = np.linspace(
        min_feasibility,
        max_feasibility,
        num=config.threshold_trials)
    # grid search
    best_auc = 0.
    best_th = -10
    val_stats = None
    with torch.no_grad():
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = model.predict_logits(val_text_rep, dataloader)
        # search the best threshold & corresponding eval results on valset by AUC
        for th in thresholds:
            temp_logits = threshold_with_feasibility(
                all_logits, val_dataset.seen_mask, threshold=th, feasibility=unseen_scores)
            results = test(val_dataset, evaluator, temp_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)
            auc = results['AUC']
            if auc > best_auc:
                best_auc = auc
                best_th = th
                if print_info:
                    print('New best AUC: ', best_auc, 'Threshold: ', best_th)
                val_stats = copy.deepcopy(results)
    return best_th, val_stats


def eval_valset(model, val_text_rep, val_dataset, config, feasibility=None, print_info=True):
    # instantiate an evaluator
    evaluator = Evaluator(val_dataset)
    drop_last = False
    dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, \
                            drop_last=drop_last, num_workers=config.num_workers, pin_memory=config.pin_memory)

    val_stats = None
    if config.open_world and config.threshold is None:
        # search the feasibility threshold
        best_th, val_stats = search_feasibility_threshold(model, val_text_rep, val_dataset, dataloader, \
            feasibility, evaluator, print_info=print_info)
    else:
        best_th = config.threshold
    
    if val_stats is None:
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt = model.predict_logits(val_text_rep, dataloader)
            if config.open_world:
                if print_info: print('using threshold: ', best_th)
                all_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=best_th, feasibility=feasibility)
            results = test(val_dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)
        val_stats = copy.deepcopy(results)
    
    # print evaluation results
    if print_info:
        result = ""
        for key in val_stats:
            result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
        print(result)

    return val_stats, best_th


def eval_testset(model, test_text_rep, test_dataset, config, best_th=None, feasibility=None):
    evaluator = Evaluator(test_dataset)
    dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

    with torch.no_grad():
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = model.predict_logits(test_text_rep, dataloader)
        if config.open_world and best_th is not None:
            print('using threshold: ', best_th)
            all_logits = threshold_with_feasibility(
                all_logits,
                test_dataset.seen_mask,
                threshold=best_th,
                feasibility=feasibility)
        test_stats = test(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )

        result = ""
        for key in test_stats:
            result = result + key + "  " + \
                str(round(test_stats[key], 4)) + "| "
        print(result)
    return test_stats


def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasibility=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasibility (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasibility >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score


def test(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, bias=config.bias, topk=config.topk
    )

    attr_acc = float(torch.mean(
        (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean(
        (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=config.topk,
    )

    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

    return stats


def dump_reports(test_stats, val_stats, best_th, config):
    results = {
        'val': val_stats,
        'test': test_stats,
    }
    if best_th is not None:
        results['best_threshold'] = best_th

    if config.experiment_name != 'clip':
        name_suffix = "open.calibrated.json" if config.open_world else "closed.json"
        if getattr(config, 'soft_embeddings', None) is not None:
            result_path = config.soft_embeddings[:-2] + name_suffix
        else:
            result_path = os.path.join(os.path.dirname(config.ckpt_file), '..', 'eval_{}'.format(name_suffix))

        with open(result_path, 'w+') as fp:
            json.dump(results, fp)


def main(config, device):
    
    # setup dataset
    dataset_path = DATASET_PATHS[config.dataset]
    print('loading validation dataset')
    num_img_aug = getattr(config, 'num_aug', 0)
    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     num_aug=num_img_aug,
                                     split='compositional-split-natural',
                                     open_world=config.open_world)
    print('loading test dataset')
    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      num_aug=num_img_aug,
                                      split='compositional-split-natural',
                                      open_world=config.open_world)
    
    # load the trained model
    model = load_trained_model(config, val_dataset, device)
    
    # pre-compute the text representations as linear classifier
    val_text_rep = get_text_representations(model, val_dataset, config)
    test_text_rep = get_text_representations(model, test_dataset, config)
    
    # load feasibility scores for open-world setting
    unseen_scores = load_feasibilities(config) if config.open_world else None

    print('evaluating on the validation set')
    val_stats, best_th = eval_valset(model, val_text_rep, val_dataset, config, feasibility=unseen_scores)

    print('evaluating on the test set')
    test_stats = eval_testset(model, test_text_rep, test_dataset, config, best_th=best_th, feasibility=unseen_scores)

    # dump the evaluation results
    dump_reports(test_stats, val_stats, best_th, config)
    print("done!")



if __name__ == "__main__":

    # get input configurations
    config = get_config()

    # set the seed value
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")
    print(f"experiment name: {config.experiment_name}")

    cfg_path = os.path.join(os.path.dirname(config.ckpt_file), '..', 'config_eval.yaml')
    with open(cfg_path, 'w') as f:
        f.writelines(pformat(vars(config)))

    main(config, device)