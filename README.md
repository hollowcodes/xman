# XMAN 

## xman is a simple python-package to manage and keep track of your (Pytorch-) deep-learning experiments

## | installation using pip
```bash
pip install xman
```

## | standart usage
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

        ...
        predict()
        backward()
        step()

        ...

    ...

    # log current epoch
    xmanager.log_epoch(model, lr, batch_size, epoch_train_accuracy, 
            epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

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

# create new experiment (give name with 'experiment_name' and tell to create with 'new=True')
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

        ...

        predict()
        backward()
        step()

        ...

    ...

    # log current epoch
    xmanager.log_epoch(model, lr, batch_size, epoch_train_accuracy, 
            epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

# plot logged training-history if wanted
xmanager.plot_history(save=True)
```
