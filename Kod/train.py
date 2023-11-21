from models.Unet import UNet
from dataset.data import BatchMaker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import datetime
import yaml


timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/noisy_labels_trainer_{}'.format(timestamp))
epoch_number = 0
EPOCHS = 5
BATCH = 1
best_vloss = 1_000_000.
path_to_config = '/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
with open(path_to_config, 'r') as config_file:
    config = yaml.safe_load(config_file)


batch_maker = BatchMaker(config_path=path_to_config, batch_size=BATCH,mode ='train')
train_loader = batch_maker.train_loader
val_loader = batch_maker.val_loader



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    batches = len(train_loader)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if batch_idx % batches == batches - 1 or (batch_idx + 1) % 10 == 0:
            last_loss = running_loss / batches # loss per batch
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + batch_idx + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")


model = UNet(3,1)
model.to(device)
loss_fn = nn.BCELoss()

# Definicja optymalizatora (np. Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for batch_idx, (vinputs, vlabels) in enumerate(val_loader):
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (batch_idx + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    scheduler.step(avg_vloss)

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = config['save_model_path'] + '/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)


    epoch_number += 1