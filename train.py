#
import os
import pprint
from pprint import pformat

import numpy as np
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from models.losses import GenCSPLoss
from models.scheduler import CustomCosineAnnealingWarmupRestarts
from utils import set_seed, get_config
from evaluate import eval_valset, get_text_representations, load_feasibilities
from tensorboardX import SummaryWriter

DIR_PATH = os.path.dirname(os.path.realpath(__file__))



def get_lr_scheduler(optimizer, config):
    lr_schedule = getattr(config, 'lr_schedule', 'step')
    if lr_schedule == 'step':
        num_decays = getattr(config, 'num_decays', 5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.epochs // num_decays, gamma=0.5)
    elif lr_schedule == 'warmup_cos':
        scheduler = CustomCosineAnnealingWarmupRestarts(optimizer=optimizer, 
                                                        warmup_epochs=10, 
                                                        T_0=config.epochs,  # no restart
                                                        eta_min=config.lr * 0.001)
    return scheduler, lr_schedule



def get_loss_fn(model, config):
    group_cfg = {'w_attr': config.w_attr, 'w_obj': config.w_obj} if getattr(config, 'with_group', False) else None
    use_gauss = getattr(config, 'use_gauss', False)
    use_attrobj_gauss = getattr(config, 'use_attrobj_gauss', False)
    group_gauss = getattr(config, 'group_gauss', False)
    ls = getattr(config, 'label_smooth', 0.0)
    partial_smooth = getattr(config, 'partial_smooth', False)
    disentangle = getattr(config, 'disentangle', False)
    if disentangle: group_cfg.update({'w_indep': getattr(config, 'w_indep', [0.0, 0.0])})
    loss_fn = GenCSPLoss(use_gauss=use_gauss, 
                            use_attrobj_gauss=use_attrobj_gauss, 
                            group_gauss=group_gauss,
                            group_cfg=group_cfg, 
                            disentangle=disentangle, 
                            ls=ls, 
                            partial_smooth=partial_smooth)
    return loss_fn, group_cfg



def evaluate(model, val_dataset, config, open_world=True):
    model.eval()
    val_text_rep = get_text_representations(model, val_dataset, config)
    # compute feasibility scores for unseen in open-world setting
    unseen_scores = load_feasibilities(config) if open_world else None
    val_stats, best_th = eval_valset(model, val_text_rep, val_dataset, config, feasibility=unseen_scores, print_info=False)
    # print results
    result = ""
    filtered_stats = dict()
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in val_stats:
        if key in key_set:
            result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
            filtered_stats[key] = val_stats[key]
    print(result)
    model.train()
    return filtered_stats


def save_the_latest(data, ckpt_file, topK=3, ignores=[]):
    """ Only keeping the latest topK checkpoints.
    """
    # find the existing checkpoints in a sorted list
    folder = os.path.dirname(ckpt_file)
    num_exist = len(os.listdir(folder))
    if num_exist >= topK + len(ignores):
        # remove the old checkpoints
        ext = ckpt_file.split('.')[-1]
        all_ckpts = list(filter(lambda x: x.endswith('.' + ext), os.listdir(folder)))
        all_epochs = [int(filename.split('.')[-2].split('_')[-1]) for filename in all_ckpts]
        fids = np.argsort(all_epochs)  # model_5.pth
        # iteratively remove
        for i in fids[:(num_exist - topK + 1)]:
            if all_epochs[i] in ignores:
                continue
            file_to_remove = os.path.join(folder, all_ckpts[i])
            if os.path.isfile(file_to_remove):
                os.remove(file_to_remove)
    torch.save(data, ckpt_file)


def train_epoch(model, train_dataloader, train_pairs, i, loss_fn, optimizer, config, group_cfg=None, writer=None):
    num_batch = len(train_dataloader)
    progress_bar = tqdm.tqdm(
        total=num_batch, desc="epoch % 3d" % (i + 1), ncols=0
    )

    epoch_train_losses = []
    for bid, batch in enumerate(train_dataloader):
        if 'debug' in config.config and bid > 1:
            break
        batch_img, batch_target = batch[0], batch[3]
        batch_target = batch_target.to(model.device)
        batch_img = batch_img.to(model.device, non_blocking=True)
        if not config.experiment_name in ['gencsp']:
            batch_img = model.encode_image(batch_img)

        outputs = model(batch_img, train_pairs)

        attr_target, obj_target = None, None
        if group_cfg is not None:
            attr_target, obj_target = batch[1].to(model.device), batch[2].to(model.device)
        losses = loss_fn(outputs, batch_target, attr=attr_target, obj=obj_target)

        # normalize loss to account for batch accumulation
        for k, v in losses.items():
            losses[k] = v / config.gradient_accumulation_steps

        # backward pass
        losses['total_loss'].backward()

        # weights update
        if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                (bid + 1 == num_batch):
            optimizer.step()
            optimizer.zero_grad()
        

        epoch_train_losses.append(losses['ce_loss'].item() if 'ce_loss' in losses else losses['total_loss'].item())
        progress_bar.set_postfix(
            {"train loss": np.mean(epoch_train_losses[-50:])}  # moving average
        )
        progress_bar.update()
        
        if writer is not None:
            # tensorboard writer
            for k, v in losses.items():
                writer.add_scalars('train/{}'.format(k), {k: v}, i * num_batch + bid)
            if isinstance(outputs, dict) and 'tau_inv' in outputs: 
                writer.add_scalar('train/tau_inv', outputs['tau_inv'], i * num_batch + bid)
            writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], i * num_batch + bid)

    progress_bar.close()
    progress_bar.write(
        f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
    )


def train_model(model, optimizer, train_dataset, val_dataset, config, device, ckpt_dir, writer=None):
    """Function to train the model to predict attributes with cross entropy loss.

    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        train_dataset (CompositionDataset): the train dataset
        val_dataset (CompositionDataset): the validation dataset
        config (argparse.ArgumentParser): the config
        device (...): torch device
        ckpt_dir: directory to save model checkpoints
    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    model.train()
    best_model_metric = getattr(config, 'best_model_metric', 'best_unseen')
    best_metric = 0
    rm_ignore_epoch = []
    keep_epoch = getattr(config, 'keep_epoch', [])

    # setup loss function
    loss_fn, group_cfg = get_loss_fn(model, config)
    #
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(device)
    if getattr(loss_fn, 'partial_smooth', False):
        loss_fn.cross_entropy_ao.comp_pairs = train_pairs
    
    torch.autograd.set_detect_anomaly(True)
    scheduler, sch_name = get_lr_scheduler(optimizer, config)
    optimizer.zero_grad()

    for i in range(config.epochs):
        # train for one epoch
        train_epoch(model, train_dataloader, train_pairs, i, loss_fn, optimizer, config, group_cfg=group_cfg, writer=writer)
        if sch_name == 'step':
            scheduler.step()
        else: 
            scheduler.step(epoch=i)

        if (i + 1) % config.save_every_n == 0:
            print("Evaluating val dataset:")
            val_result = evaluate(model, val_dataset, config)
            # update the best val metric
            if val_result[best_model_metric] > best_metric:
                best_metric = val_result[best_model_metric]
                rm_ignore_epoch = [i+1] + keep_epoch
            
            if writer is not None:
                # write to tensorboards
                for k, v in val_result.items():
                    writer.add_scalars('val/{}'.format(k), {k: v}, (i+1)*len(train_dataloader))
            
            # save model
            save_dict = {'epoch': i+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            ckpt_file = os.path.join(ckpt_dir, 'model_{}.pt'.format(i+1))
            save_the_latest(save_dict, ckpt_file, topK=1, ignores=rm_ignore_epoch)
            print('Model has been saved: {}\n'.format(ckpt_file))

    return model, best_metric


if __name__ == "__main__":
    # get input configurations
    config = get_config()

    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("==> training details: ")
    pprint.pprint(vars(config))
    num_img_aug = getattr(config, 'num_aug', 0)
    if num_img_aug > 0:
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    use_tb = getattr(config, 'use_tb', True)
    writer = None
    if use_tb:
        # tensorboard path
        logs_dir = os.path.join(config.save_path, 'tensorboards')
        os.makedirs(logs_dir, exist_ok=True)
        writer = SummaryWriter(logs_dir)

    # checkpoints path
    ckpt_dir = os.path.join(config.save_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       num_aug=num_img_aug,
                                       split='compositional-split-natural')
    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     num_aug=num_img_aug,
                                     split='compositional-split-natural')
    
    model, optimizer = get_model(train_dataset, config, device)

    print("model dtype", model.dtype)
    print("soft embedding dtype", model.soft_embeddings.dtype)

    with open(os.path.join(config.save_path, 'config_train.yaml'), 'w') as f:
        f.writelines(pformat(vars(config)))

    model, best_metric = train_model(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        config,
        device,
        ckpt_dir,
        writer=writer
    )

    if use_tb:
        writer.close()
    print("done!")
