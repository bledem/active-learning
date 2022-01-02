# Uncertainty sampling

We evaluate the uncertainty of the model with different sampling techniques ['ratio', 'margin', 'least', 'entropy', 'random'].

CNN and ResnetX have been implemented. We use wanb to monitor the training.
You can use CIFAR-10, EMNIST or MNIST data.

## Experiments

### Training
We train an initial model on 50% of the original dataset.
Pretending that 50% of the dataset is hidden with the parameter split_train_val_ratio=0.5.

```
python3 training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```

```
python3 training/run_experiment.py --model_class=Resnet18 --data_class=CIFAR10DataModule --max_epochs=15 --gpus=1
```

(if you use CNN you need to set manully the image size as global variable IMAGE_SIZE to 32 for CIFAR10 and 28 for EMNIST)

### Estimation of uncertainty
We apply the uncertain scoring techniques and save the highest uncertainty images. Change the arguments __main__. 

```
python3 training/create_new_set.py --model_class=Resnet18 --data_class=CIFAR10DataModule 
```


```
python3 training/create_new_set.py --model_class=Resnet18 --data_class=CIFAR10DataModule 
```
