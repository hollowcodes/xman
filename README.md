# XMAN 

## xman is a simple python-package to manage and keep track of your (Pytorch-) deep-learning experiments

## | description
### xman creates a 'x-manager/' folder in a given directory, which contains experiment-folders which each have a json-file with the hyperparameters and results of an experiment.

## | installation using pip
```bash
pip install xman
```

## | x-manager/ structure
### the name of the experiment-folders must be chosen by the user.
```
your_directory/
    ––– model.py
    --- train.py
    --- dataset.py
    --- preprocessing.py
    
    --- x-manager/
            --- experiment1/
                    --- train_stats.json
                    --- model.pt
                    --- train_history.png
            --- experiment2/
                    --- train_stats.json
                    --- model.pt
                    --- train_history.png
            . . .
```

## | train_stats.json
### 'learning-rate' and batch-size' are also lists, because they can change while training (ie. learning-rate decay).
```json
{
    "hyperparameters": {
        "optimizer": "Adam",
        "loss-function": "MSE",
        "epochs": 10,
        "learning-rate": [
            0.001
        ],
        "batch-size": [
            256
        ]
    },
    "results": {
        "train-accuracy": [
            12.7551,
            11.12245,
            . . .
        ],
        "train-loss": [
            2.310244143009186,
            2.3060443997383118,
            . . .
        ],
        "validation-accuracy": [
            20.0,
            20.0,
            . . .
        ],
        "validation-loss": [
            2.304267168045044,
            2.2817447185516357,
            . . .
        ]
    }


```

## | standart usage

### To create a new experiment, set 'new=True' after the name of the experiment. Before running the experiment the next time this has to be set back to 'False', otherwise it will be recreated and the saved stats will be deleted.
### If you interrupt the trainings-process (ie. to change the learning-rate), but you want to stay in the same training-session (by loading the last checkpoint) set 'continue_=True'. Otherwise the saved training-stats of the experiment will be overwritten, when restarting the training.
```python
import torch
from xman import ExperimentLogger

# functions
optimizer = torch.optim.Adam()
criterion = torch.nn.MSELoss()

# hyperparameters
epochs = 20
lr = 0.001
batch_size = 64

# create new experiment (give name with 'experiment_name' and tell to create with 'new=True')
xmanager = ExperimentManager(experiment_name="experiment1", new=True, continue_=False)

# initialize xmanager with (initial) hyperparameters
xmanager.init(optimizer="Adam", 
              loss_function="MSELoss", 
              epochs=epochs, 
              learning_rate=lr, 
              batch_size=batch_size)

# train loop
for epoch in range(epochs):

    # data loop
    for sample, target in dataset:

        . . .

        predict()
        backward()
        step()

        . . .

    . . .

    # log current epoch
    xmanager.log_epoch(model, lr, batch_size, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

# plot logged training-history
xmanager.plot_history(save=True)
```

## | usage with additional custom hyperparameters
```Python
import torch
from xman import ExperimentLogger

# functions
optimizer = torch.optim.Adam()
criterion = torch.nn.MSELoss()

# hyperparameters
epochs = 20
lr = 0.001
batch_size = 64
# additional hyperparameters
lstm_hidden_size = 256
lstm_layers = 3
droput_chance = 0.45

# create new experiment
# give name with 'experiment_name' and tell to create with 'new=True', before running it again change 'new' to False
xmanager = ExperimentManager(experiment_name="experiment1", new=True, continue_=False)

# initialize xmanager with (initial) hyperparameters
xmanager.init(optimizer="Adam", 
              loss_function="MSELoss", 
              epochs=epochs, 
              learning_rate=lr, 
              batch_size=batch_size,
              
              # dictionary with additional custom hyperparameters
              custom_parameters={
                    "lstm-hiddensize": lstm_hidden_size,
                    "lstm-layers": lstm_layers,
                    "dropout-change": dropout_chance
              })

# train loop
for epoch in range(epochs):

    # data loop
    for sample, target in dataset:

        . . .

        predict()
        backward()
        step()

        . . .

    . . .

    # log current epoch
    xmanager.log_epoch(model, lr, batch_size, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

# plot logged training-history if wanted
xmanager.plot_history(save=True)
```
