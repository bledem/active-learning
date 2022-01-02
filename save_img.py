import h5py
from PIL import Image
from pathlib import Path
import os 
import matplotlib.pyplot as plt
import numpy as np 

EMNIST_MAP = ['<B>', '<S>', '<E>', '<P>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']
CIFAR10_MAP = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

LABEL_MAP = CIFAR10_MAP

def open_h5(path):
    with h5py.File(path, "r") as f:
        x_trainval = f["x_train"][:]
        y_trainval = f["y_train"][:].squeeze().astype(int)
    return x_trainval, y_trainval

def save_imgs(x_list, y_list, save_dir):
    for i, img in enumerate(x_list):
        img = Image.fromarray(img*255).convert('RGB')
        img.save(str(save_dir/ f'{i}_label_{LABEL_MAP[y_list[i]]}.png'))

def save_single_img(x_list, y_list, save_dir):
    num_row = 10
    num_col = 10# plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(len(x_list)):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(x_list[i], cmap='gray')
        ax.set_title('Label: {}'.format(LABEL_MAP[y_list[i]]))
    plt.tight_layout()
    plt.savefig(str(save_dir/'all.png'))

if __name__ == '__main__':
    # EMNIST
    org_dataset = '/home/bettyld/PJ/Documents/data/processed/emnist/byclass.h5'
    # CIFAR-10
    org_dataset = '/home/bettyld/PJ/Documents/active-learning/sampled/CIFAR10DataModule/Resnet18/5hr9s5nt/checkpoints/epoch=006-val_loss=0.890-val_cer=0.000/least/reasonable_dataset_uncertain_dataset.h5'

    checkpoint_dir = Path('/home/bettyld/PJ/Documents/active-learning/sampled/EMNIST/CNN/ubq73h2h/checkpoints/epoch=006-val_loss=0.541-val_cer=0.000')
    checkpoint_dir = Path('/home/bettyld/PJ/Documents/active-learning/sampled/CIFAR10DataModule/Resnet18/5hr9s5nt/checkpoints/epoch=006-val_loss=0.890-val_cer=0.000')
   
    sampling_technique_list = ['ratio', 'margin', 'least', 'entropy', 'random']
    for technique in sampling_technique_list:
        print('technique:',technique)
        sampled_dataset = checkpoint_dir / technique / 'uncertain_dataset.h5'
  
        save_dir_ = checkpoint_dir / technique /'imgs'
        save_dir = save_dir_ / 'org'
        x_org, y_org = open_h5(org_dataset)
        x_s, y_s = open_h5(sampled_dataset)

        os.makedirs(save_dir, exist_ok=True)
        idx_list  = np.random.randint(0, len(x_org), 100)
        # save_imgs(x_org[idx_list], y_org[idx_list], save_dir)
        save_single_img(x_org[idx_list], y_org[idx_list], save_dir)

        save_dir = save_dir_ / 'sampled'
        idx_list  = np.random.randint(0, len(x_s), 100) 
        os.makedirs(save_dir, exist_ok=True)
        # save_imgs(x_s[idx_list], y_s[idx_list], save_dir)
        save_single_img(x_s[idx_list], y_s[idx_list], save_dir)

    # x_s, y_s = open_h5(sampled_dataset_only)
    # idx_list  = np.random.randint(0, len(x_s), 1000)
    # save_dir_ = save_dir / 'sampled_only'
    # os.makedirs(save_dir_, exist_ok=True)
    # save_imgs(x_s[idx_list], y_s[idx_list], save_dir_)
    # save_single_img(x_s[idx_list], y_s[idx_list], save_dir_)

    