"""Experiment-running framework.
This script creates:
- A mix of hard sampled + original training set  ``rand_merged_{trained_ratio}_plus_uncertain_dataset.h5``
- A mix of medium (neither high or low loss) + training set ``reasonable_dataset_uncertain_dataset``
- Save low and high loss sample
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn import Softmax
from torch.utils.data import Subset
import h5py
import os 
import tqdm
from PIL import Image
import math
import json

from text_recognizer.data.base_data_module import BaseDataModule
from text_recognizer import lit_models



# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "active_batch" / "emnist"
TO_LABEL_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "batch_1.h5"

EMNIST_MAP = ['<B>', '<S>', '<E>', '<P>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']
CIFAR10_MAP = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
org_dataset = PROCESSED_DATA_DIRNAME / "byclass.h5"
LABEL_MAP = CIFAR10_MAP


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="EMNIST")
    parser.add_argument("--model_class", type=str, default="CNN")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--split_train_val_ratio", type=float, default=0.5)
    parser.add_argument("--data_path", type=str, default=str(org_dataset))

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def predict_unseen(model, data_list, return_type='softmax'):
    preds_logits, pred_labels = [], []
    softmax = Softmax()
    for data_pt, label in tqdm.tqdm(data_list):
        batch_data_pt = data_pt.unsqueeze(0)
        pred = logit = model(batch_data_pt)
        if return_type == 'softmax' or return_type == 'argmax':
            pred = softmax(logit)
        _, pred_arg = torch.topk(pred, 1)
        pred_arg = pred_arg[0]
        # pred = np.array(pred[0])
        pred = pred[0]
        preds_logits.append(pred)
        pred_labels.append(pred_arg)
    return preds_logits, pred_labels

def least_confidence(prob_dist):
    # from human in the loop ML book
    """
    Return the uncertainty score of an array using least confidence sampling 
    in 0-1 range where 1 is the more uncertain.
    Assumes prob tensor is like tensor([0.1 0.7 0.2])
    """
    simple_least_conf = torch.max(prob_dist)
    num_labels = prob_dist.numel()                   
    normalized_least_conf = (1- simple_least_conf) * (num_labels /
    (num_labels -1))                     
    return normalized_least_conf.item()   
    
def margin_confidence(prob_dist):
    """Return the uncertainty score of a probability distribution using a margin
        of confidence in 0-1 range where 1 is most uncertain. 
        margin_confidence([0.0321 0.6439 0.0871 0.2369]) = 0.5930
    """
    prob_dist, _ = torch.sort(prob_dist, descending=True)
    diff = (prob_dist.data[0] - prob_dist[1])
    margin_conf = 1 - diff
    return margin_conf.item()

def ratio_confidence(prob_dist):
    """
    ratio_confidence([0.0321 0.6439 0.0871 0.2369]) = 0.5930
    """
    prob_dist, _ = torch.sort(prob_dist, descending=True)
    ratio_conf = prob_dist.data[1] / prob_dist.data[0]
    return ratio_conf.item()

def entropy_score(prob_dist):
    """
    Return uncertainty score for a probability distribution utorchsing entropy
    entropy_score([0.0321 0.6439 0.0871 0.2369]) = 0.684

    """
    log_probs = prob_dist * torch.log2(prob_dist)
    raw_entropy = - torch.sum(log_probs)
    normalized_entropy = raw_entropy / math.log2(len(prob_dist))
    return normalized_entropy.item()

def random_score():
    rands = np.random.uniform(0, 1)
    return rands


def rank_preds(list_preds, sampling_type='least'):
    uncertainty_scores = []
    for pred_array in list_preds:
        if sampling_type=='least':
            # take the difference between 100% and the most confident label
            ranked_preds = least_confidence(pred_array)
        elif sampling_type=='margin':
            ranked_preds = margin_confidence(pred_array)
        elif sampling_type=='ratio':
            ranked_preds = ratio_confidence(pred_array) 
        elif sampling_type=='entropy':
            ranked_preds = entropy_score(pred_array) 
        elif sampling_type=='random':
            ranked_preds = random_score()
        uncertainty_scores.append(ranked_preds)
    return uncertainty_scores


def save_sample_img(save_dir, results, preds, un_score, type=None, max=20):
    img_tensor = extract_tensor(results)
    labels = results[1]
    for i, x in enumerate(img_tensor[:max]):
        y = LABEL_MAP[labels[i]]
        p = LABEL_MAP[preds[i][0].item()]
        sc = un_score[i]
        img_name = save_dir.joinpath(f'{type}_pred_{p}_y_{y}_sc_{sc}.png')
        # display friendly
        x = np.array(x*255, dtype='uint8')
        im = Image.fromarray(x).convert('RGB')
        im.save(str(img_name))

def save_dataset(save_path, x_train, y_train, x_val, y_val, x_test, y_test):

    with h5py.File(save_path, "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_test", data=x_test)
        f.create_dataset("y_test", data=y_test)
        if x_val is not None:
            f.create_dataset("x_val", data=x_val)
            f.create_dataset("y_val", data=y_val)


def sample(uncertainty_scores, list_labels, data, n_samples, rev=True):
    sorted_uncertainty_scores_idx = np.argsort(uncertainty_scores) # from lowest to highest uncertainty score 
    # Sampling 
    if rev:
        size = -1
    else:
        size = 1
    topk_uncertainty_scores_idx = list(sorted_uncertainty_scores_idx[::size][:n_samples])
    topk_sorted_uncertainty_scores = np.array(uncertainty_scores)[topk_uncertainty_scores_idx]
    topk_sorted_uncertainty_label = np.array([e.numpy() for e in list_labels])[topk_uncertainty_scores_idx]
    topk_uncertainty_data = data.data_train[topk_uncertainty_scores_idx]
    topk_indx = data.data_train.indices[topk_uncertainty_scores_idx]
    return topk_uncertainty_data, topk_sorted_uncertainty_label, topk_sorted_uncertainty_scores, topk_indx

def write_idx_to_json(save_path: str, data: list, write_or_append='w'):
    data_to_write = {}
    data_to_write['indices'] = data
    save_file = os.path.join(save_path, 'sampled.json')
    with open(save_file, write_or_append) as outfile:
        json.dump(data_to_write, outfile)
def format_array(arr):
    arr = list(arr.astype(int))
    arr = [int(elt) for elt in arr]
    return arr

def extract_tensor(arr):
    """ Take arr a tuple of (data, tgt), we return data."""
    try:
        x_train = np.array(arr[0]).transpose(1,2,0)
    except:
        # convert tensor to array
        x_train = np.array([e.cpu().detach().numpy() for e in arr[0]])
        x_train = x_train.transpose(0,2,3,1) #(10000, 3, 32, 32) -> (10000, 32, 32, 3)
    return x_train

def run_experiment(lit_model, data, list_sampling_technique, n_samples, save_dir, trained_ratio, add_rand=None):
    # predict
    lit_model.eval()
    lit_model.freeze()
    # getting unseen data 
    # assert len(data.data_train) == len_complete_dataset

    # 1 - get unseen and seen data
    break_idx = int(len(data.data_train)*trained_ratio)
    seen_subset = Subset(data.data_train, np.arange(break_idx))
    # training
    seen_data_dict = {}
    nb_channels = len(seen_subset[0][0])
    seen_subset_data =  np.array([ np.array(seen_subset[idx][0][:nb_channels])  for idx in seen_subset.indices])
    seen_subset_targets =  np.array([ np.array(seen_subset[idx][1])  for idx in seen_subset.indices])
    seen_data_dict['training'] = [seen_subset_data, seen_subset_targets]
    # validation
    val_subset = data.data_val
    seen_subset_data =  np.array([ np.array(val_subset.dataset[idx][0][:nb_channels])  for idx in val_subset.indices])
    seen_subset_targets =  np.array([ np.array(val_subset.dataset[idx][1])  for idx in val_subset.indices])
    seen_data_dict['validation'] = [seen_subset_data, seen_subset_targets]
    # assert len(seen_data_dict['validation'][1]) == len(seen_data_dict['validation'][0]) == 65054

    unseen_range = np.arange(break_idx, len(data.data_train))
    # to make the prediction faster
    # unseen_range = np.random.choice(unseen_range, 10000, replace=False)
    data.data_train = Subset(data.data_train, unseen_range)
    
    # 2 - run the model on all samples
    list_preds, list_labels = predict_unseen(lit_model, data.data_train)

    # uncertainty sampling, sampling close from decision boundary
    # 3- Rank the predictions per uncertainty score given the sampling_technique
    for sampling_technique in list_sampling_technique:
        save_dir_technique = save_dir / sampling_technique
        # return uncertainty score for each instance
        uncertainty_scores = rank_preds(list_preds, sampling_type=sampling_technique)
        # 4 - Sample the topN most uncertain instances
        topk_data, topk_label, topk_score, topk_uncertainty_scores_idx = sample(uncertainty_scores, list_labels, data, n_samples, rev=True)
        topk_uncertainty_scores_idx = format_array(topk_uncertainty_scores_idx)
        # 5- save images
        os.makedirs(save_dir_technique, exist_ok=True)
        save_sample_img(save_dir_technique, topk_data, topk_label, topk_score, type='high')
        # save dataset
        save_dataset_path = os.path.join(save_dir_technique, 'uncertain_dataset.h5')

        #prepare & save dataset
        # topk_data[0] is (28, 10000, 28) -> x_train (batch_size, x_size, y_size)
        x_train = extract_tensor(topk_data)
        
        # topk_label (batch_size, 1) -> (batch_size, 1, 1)
        y_train = np.expand_dims(np.array(topk_label), 1)
        x_test = np.array(data.data_test.data)
        # (batch_size,) -> (batch_size,1)
        y_test = np.expand_dims(np.array(data.data_test.targets), 1)

        # x_val -> (batch_size, width, height)
        if nb_channels>1:
            x_val = seen_data_dict['validation'][0].transpose(0,2,3,1)
        else:
            x_val = seen_data_dict['validation'][0][:,0,:,:]
        y_val = np.expand_dims(seen_data_dict['validation'][1], 1)
        save_dataset(save_dataset_path, x_train, y_train, x_val, y_val, x_test, y_test)
        # exclude the most difficult samples
        write_idx_to_json(save_dir_technique, topk_uncertainty_scores_idx)
        # 6 - prepare & save the dataset (training_data+sampled instance) to load in next training
        # x shape in h5 ->(nb_instance, img_width, img_heigth)
        # y shape in h5 -> (nb_instance, 1)
        x_train_sampled = x_train
        y_train_sampled = np.array(topk_label)
        if nb_channels>1:
            x_train_org = seen_data_dict['training'][0].transpose(0,2,3,1)
        else:
            x_train_org = seen_data_dict['training'][0][:,0,:,:] # TO CHECK
        y_train_org = np.expand_dims(seen_data_dict['training'][1], 1)


        if add_rand:
            uncertainty_scores = rank_preds(list_preds, sampling_type='random')
            topk_data_s, topk_label_s, _, _ = sample(uncertainty_scores, list_labels, data, add_rand, rev=True)
            x_train_rand = extract_tensor(topk_data_s)
            # x_train_rand = np.array(topk_data_s[0]).transpose(1,2,0)
            y_train_rand = np.array(topk_label_s)
            # assert len(x_train_rand) > len(x_train_sampled)
            x_train = np.vstack([x_train_org, x_train_sampled, x_train_rand])
            y_train = np.vstack([y_train_org, y_train_sampled, y_train_rand])
            save_dataset_path = os.path.join(save_dir_technique, f'rand_merged_{trained_ratio}_plus_uncertain_dataset.h5')

        else:
            # (size, width, height) for both array below
            x_train = np.vstack([x_train_org, x_train_sampled])
            y_train = np.vstack([y_train_org, y_train_sampled])
            save_dataset_path = os.path.join(save_dir_technique, f'merged_{trained_ratio}_plus_uncertain_dataset.h5')

        save_dataset(save_dataset_path, x_train, y_train, x_val, y_val, x_test, y_test)

        # Extra 1 - Save the top N most certain (for academic purpose)
        topk_data, topk_label, topk_score, top_idx = sample(uncertainty_scores, list_labels, data, n_samples, rev=False)
        top_idx = format_array(top_idx)
        save_sample_img(save_dir_technique, topk_data, topk_label, topk_score, type='low')
        # exclude the easiest samples
        write_idx_to_json(save_dir_technique, top_idx, 'a')

        # Extra 2 - Retrieve unselected data
        reasonble_samples = list(set(unseen_range) - set(top_idx) - set(topk_uncertainty_scores_idx))
        small_reasonable_samples = np.random.choice(reasonble_samples, n_samples, replace=False)
        train_subset = data.data_train
        x_train_sampled =  np.array([ np.array(train_subset.dataset[idx][0][:nb_channels])  for idx in small_reasonable_samples])
        y_train_sampled =  np.array([ np.array(train_subset.dataset[idx][1])  for idx in small_reasonable_samples])
        y_train_sampled = np.expand_dims(y_train_sampled, 1)
        if nb_channels>1:
            x_train_sampled = x_train_sampled.transpose(0,2,3,1)
        else:
            x_train_sampled = x_train_sampled[:,0,:,:]
        x_train = np.vstack([x_train_org, x_train_sampled])
        y_train = np.vstack([y_train_org, y_train_sampled])
        save_dataset_path = os.path.join(save_dir_technique, 'reasonable_dataset_uncertain_dataset.h5')
        save_dataset(save_dataset_path, x_train, y_train, x_val, y_val, x_test, y_test)

def get_emnist():
    PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
    dataset = PROCESSED_DATA_DIRNAME / "byclass.h5"
    return dataset

def get_cifar():
    PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "cifar10"
    dataset = PROCESSED_DATA_DIRNAME 
    return dataset

def get_data(model_name):
    if model_name =='EMNIST':
        return get_emnist()
    else:
        return get_cifar()

def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    n_samples = 5000
    add_random = int(1*n_samples)
    sampling_technique_list = ['ratio', 'margin', 'least', 'entropy', 'random']

    pj_dir = Path(os.path.dirname(__file__))

    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")
    trained_ratio = args.split_train_val_ratio
    data_path = get_data(args.data_class)
    data = data_class(data_path, args, split_ratio=1)
    data.setup(split_train_val=1) # we'll take the unseen data after
    save_dir = pj_dir / 'sampled' / args.data_class / args.model_class

    model = model_class(data_config=data.config(), args=args)
    data_chkp_dir = '/home/bettyld/PJ/Documents/active-learning/training/logs/active'
    if not args.load_checkpoint:
        if data_class.__name__ == 'EMNIST': 
            # check_pt_name = "2bmy5g8j/checkpoints/epoch=005-val_loss=0.575-val_cer=0.000.ckpt"
            check_pt_name = "ubq73h2h/checkpoints/epoch=006-val_loss=0.541-val_cer=0.000.ckpt"
            # check_pt_name = "2jeh3csj/checkpoints/epoch=005-val_loss=0.533-val_cer=0.000.ckpt"
            args.load_checkpoint = os.path.join(data_chkp_dir, check_pt_name)
            len_complete_dataset = 260212 # 5000
        elif data_class.__name__ == 'MNIST':
            check_pt_name = "203lx9ac/checkpoints/epoch=003-val_loss=0.069-val_cer=0.000.ckpt"
            args.load_checkpoint = os.path.join(data_chkp_dir, check_pt_name)
            len_complete_dataset = 55000
        elif data_class.__name__== 'CIFAR10DataModule':
            # cnn
            # check_pt_name = 'iog28td9/checkpoints/epoch=006-val_loss=0.958-val_cer=0.000.ckpt'    
            check_pt_name = '5hr9s5nt/checkpoints/epoch=006-val_loss=0.890-val_cer=0.000.ckpt'
            len_complete_dataset = 45000
            args.load_checkpoint = os.path.join(data_chkp_dir, check_pt_name)
    save_dir = save_dir / check_pt_name.strip('.ckpt')
    assert len(data.data_train) == len_complete_dataset, len(data.data_train)
    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        raise ValueError('No checkpoint found')

    run_experiment(lit_model, data, sampling_technique_list, n_samples, save_dir, trained_ratio, add_rand=add_random)

    

if __name__ == "__main__":
    main()