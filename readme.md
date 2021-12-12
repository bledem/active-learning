# Experiment

Pretending that 50% of the dataset is hidden, what is the performance?

Increase in performance after adding randomly the data by 10% chunk.
Adding 10% remaining by activate learning iteration.

Activate learning 

training MNIST
```
python3 training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```

training EMNIST 
```
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1
```

Mnist dataset uses 60,000 labeled images