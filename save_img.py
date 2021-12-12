import h5py
from PIL import Image
from pathlib import Path
import os 
import matplotlib.pyplot as plt
import numpy as np 

EMNIST_MAP = ['<B>', '<S>', '<E>', '<P>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']

def open_h5(path):
    with h5py.File(path, "r") as f:
        x_trainval = f["x_train"][:]
        y_trainval = f["y_train"][:].squeeze().astype(int)
    return x_trainval, y_trainval

def save_imgs(x_list, y_list, save_dir):
    for i, img in enumerate(x_list):
        img = Image.fromarray(img*255).convert('RGB')
        img.save(str(save_dir/ f'{i}_label_{EMNIST_MAP[y_list[i]]}.png'))

def save_single_img(x_list, y_list, save_dir):
    num_row = 100
    num_col = 10# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(len(x_list)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(x_list[i], cmap='gray')
        ax.set_title('Label: {}'.format(EMNIST_MAP[y_list[i]]))
    plt.tight_layout()
    plt.savefig(str(save_dir/'all.png'))

if __name__ == '__main__':
    sampled_dataset = '/home/bettyld/PJ/Documents/active-learning/sampled/3ffq58ne/checkpoints/epoch=004-val_loss=0.597-val_cer=0.000/entropy/merged_0.25_plus_uncertain_dataset.h5'
    sampled_dataset_only = '/home/bettyld/PJ/Documents/active-learning/sampled/3ffq58ne/checkpoints/epoch=004-val_loss=0.597-val_cer=0.000/entropy/uncertain_dataset.h5'
   
    sampled_dataset = '/home/bettyld/PJ/Documents/active-learning/sampled/2bmy5g8j/checkpoints/epoch=005-val_loss=0.575-val_cer=0.000/entropy/rand_merged_0.25_plus_uncertain_dataset.h5'
    sampled_dataset_only = '/home/bettyld/PJ/Documents/active-learning/sampled/2bmy5g8j/checkpoints/epoch=005-val_loss=0.575-val_cer=0.000/entropy/uncertain_dataset.h5'
   
    org_dataset = '/home/bettyld/PJ/Documents/data/processed/emnist/byclass.h5'

    save_dir = Path('/home/bettyld/PJ/Documents/active-learning/data/vis_dataset_resnet18')
    
    save_dir_ = save_dir / 'org'
    x_org, y_org = open_h5(org_dataset)
    x_s, y_s = open_h5(sampled_dataset)

    # os.makedirs(save_dir_, exist_ok=True)
    # idx_list  = np.random.randint(0, len(x_s), 1000)
    # save_imgs(x_org[idx_list], y_org[idx_list], save_dir_)
    # save_single_img(x_org[idx_list], y_org[idx_list], save_dir_)

    # save_dir_ = save_dir / 'sampled'
    # os.makedirs(save_dir_, exist_ok=True)
    # save_imgs(x_s[idx_list], y_s[idx_list], save_dir_)
    # save_single_img(x_s[idx_list], y_s[idx_list], save_dir_)

    x_s, y_s = open_h5(sampled_dataset_only)
    idx_list  = np.random.randint(0, len(x_s), 1000)
    save_dir_ = save_dir / 'sampled_only'
    os.makedirs(save_dir_, exist_ok=True)
    save_imgs(x_s[idx_list], y_s[idx_list], save_dir_)
    save_single_img(x_s[idx_list], y_s[idx_list], save_dir_)

    