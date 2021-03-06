"""Experiment-running framework."""
import argparse
import importlib
import os 

import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger  # newline 1
import wandb
from text_recognizer import lit_models

from text_recognizer.data.base_data_module import  BaseDataModule


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


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
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="CNN")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--split_train_val_ratio", type=float, default=0.5)
    parser.add_argument("--technique", type=str, default='original')
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

def get_emnist(technique):
    PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
    # SAMPLE_DATA_DIR = Path("/home/bettyld/PJ/Documents/active-learning/sampled/3ffq58ne/checkpoints/epoch=004-val_loss=0.597-val_cer=0.000")
    SAMPLE_DATA_DIR = Path('/home/bettyld/PJ/Documents/active-learning/sampled/2jeh3csj/checkpoints/epoch=005-val_loss=0.533-val_cer=0.000')
    # sampled_dataset =  SAMPLE_DATA_DIR / technique / 'rand_merged_0.25_plus_uncertain_dataset.h5'
    # sampled_dataset =  SAMPLE_DATA_DIR / technique / 'uncertain_dataset.h5'
    if technique=='original':
        dataset = PROCESSED_DATA_DIRNAME / "byclass.h5"
    else:
        dataset = SAMPLE_DATA_DIR / technique / 'reasonable_dataset_uncertain_dataset.h5' 
    return dataset

def get_cifar(technique):
    PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "cifar10"
    SAMPLE_DATA_DIR = Path('/home/bettyld/PJ/Documents/active-learning/sampled/CIFAR10DataModule/CNN/iog28td9/checkpoints/epoch=006-val_loss=0.958-val_cer=0.000')
    SAMPLE_DATA_DIR = Path('/home/bettyld/PJ/Documents/active-learning/sampled/CIFAR10DataModule/Resnet18/5hr9s5nt/checkpoints/epoch=006-val_loss=0.890-val_cer=0.000')
    if technique=='original':
        dataset = PROCESSED_DATA_DIRNAME 
    else:
        dataset = SAMPLE_DATA_DIR / technique / 'rand_merged_0.5_plus_uncertain_dataset.h5' 
        # dataset = SAMPLE_DATA_DIR / technique / 'reasonable_dataset_uncertain_dataset.h5' 
    return dataset


def get_data(model_name, technique):
    if model_name =='EMNIST':
        return get_emnist(technique)
    else:
        return get_cifar(technique)


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")
    technique = args.technique
    data_path = get_data(args.data_class, technique)
    data = data_class(data_path, args, split_ratio=args.split_train_val_ratio)
    data.setup()
    model = model_class(data_config=data.config(), args=args)
    monitor_metric = 'val_acc'
    monitor_mode = 'max'

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseLitModel
    data_chkp_dir = '/home/bettyld/PJ/Documents/active-learning/training/logs/active'

    # args.load_checkpoint = os.path.join(data_chkp_dir, check_pt_name)
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    data_config = {'model': args.model_class,'sampling': technique,'data_path': data_path,'data_ratio': data.split_train_val_ratio,
     'len_train': len(data.data_train), 'len_data_val': len(data.data_val), 'len_test': len(data.data_test)}
    logger = pl.loggers.TensorBoardLogger("training/logs")
    wandb.init(project="active", config=data_config) 
    logger = WandbLogger() 
    wandb.run.name = f'{args.data_class}_{technique}_{data.split_train_val_ratio}_{args.model_class}'

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor=monitor_metric, mode=monitor_mode
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    # pylint: disable=no-member
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member
    print(model_checkpoint_callback.best_model_path)



if __name__ == "__main__":
    main()

    # /home/bettyld/PJ/Documents/active-learning/training/logs/active/3ffq58ne/checkpoints/epoch=004-val_loss=0.597-val_cer=0.000.ckpt