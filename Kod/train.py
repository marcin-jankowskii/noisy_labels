from models.Unet import UNet
from dataset.data import BatchMaker
from utils.metrics import SegmentationMetrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import datetime
import yaml
import matplotlib.pyplot as plt
import kornia as K
import numpy as np
import wandb


timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/noisy_labels_trainer_{}'.format(timestamp))
epoch_number = 0
num_classes = 3
EPOCHS = 100
BATCH = 6
learning_rate = 0.0001
best_viou = 1_000_000.
#path_to_config = '/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
path_to_config = '/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml'
#path_to_config = '/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml'
with open(path_to_config, 'r') as config_file:
    config = yaml.safe_load(config_file)


batch_maker = BatchMaker(config_path=path_to_config, batch_size=BATCH,mode ='train',segment = 'mixed',annotator= 2)
train_loader = batch_maker.train_loader
val_loader = batch_maker.val_loader


class MyAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # we define and cache our operators as class members
        self.k1 = K.augmentation.ColorJitter(0.15, 0.25, 0.25, 0.25)
        self.k2 = K.augmentation.RandomAffine([-45.0, 45.0], [0.0, 0.15], [0.5, 1.5], [0.0, 0.15])

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. apply color only in image
        # 2. apply geometric tranform
        img_out = self.k2(self.k1(img))

        # 3. infer geometry params to mask
        # TODO: this will change in future so that no need to infer params
        mask_out = self.k2(mask, self.k2._params)

        return img_out, mask_out



def train_one_epoch(epoch_index, tb_writer,augementation, T_aug = False,div =0):
    running_loss = 0.
    last_loss = 0.
    batches = len(train_loader)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for batch_idx, (inputs, labels,ids) in enumerate(train_loader):
        # Every data instance is an input + label pair

        div += 1
        if T_aug == True:
            for i in range(inputs.shape[0]):
                inputs[i], labels[i] = augementation(inputs[i], labels[i])


        inputs = inputs.to(device)
        labels = labels.to(device)
        ids = ids.to(device)


        #fig, ax = plt.subplots(1, 6, figsize=(20, 10))
        #for i in range(inputs.shape[0]):
            #ax[i].imshow(labels[i].cpu().numpy().transpose(1,2,0))
            #ax[i].contour(labels[i].cpu().numpy().transpose(1,2,0).squeeze(), colors='k', levels=[0.5])
        #plt.show()

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)


        #print('Output = '+str(outputs.shape))
        #print('Labels = ' + str(ids.unique()))
        #print('Labels_shape = ' + str(ids.shape))
        # Compute the loss and its gradients
        loss = loss_fn(outputs, ids)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
     

        # Gather data and report
        running_loss += loss.item()
        if batch_idx % batches == batches - 1 or (batch_idx + 1) % 10 == 0:
            last_loss = running_loss / div # loss per batch
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            #tb_x = epoch_index * len(train_loader) + batch_idx + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            #plt.subplot(1,2,1)
            #plt.imshow(outputs[0].detach().cpu().numpy().transpose(1,2,0))
            #plt.subplot(1,2,2)
            #plt.imshow(labels[0].detach().cpu().numpy().transpose(1,2,0))
            #plt.pause(0.05)
            div = 0

    return last_loss


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")


model = UNet(3,num_classes)
model.to(device)

# Binary semantic segmentation problem
#loss_fn = nn.BCELoss()
# Multi-class semantic segmentation problem
# Assume `num_classes` is the number of classes
weights = torch.ones(num_classes)

# Set a higher weight for the second class
# weights[0] = 0.1
# weights[1] = 0.7
# weights[2] = 0.4

# If you're using a GPU, move the weights tensor to the same device as your model
weights = weights.to(device)

loss_fn = nn.CrossEntropyLoss(weight=weights)

aug = MyAugmentation()

# Definicja optymalizatora (np. Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

wandb.init(project="noisy_labels", entity="segsperm")
wandb.run.name = writer
wandb.watch(model, log="all")



for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer,aug,T_aug = False)

    running_vloss = 0.0
    running_viou = 0.0
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for batch_idx, (vinputs, vlabels,vids) in enumerate(val_loader):
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            vids = vids.to(device)
            voutputs = model(vinputs)
            preds = torch.argmax(voutputs, dim=1)  # assuming a classification task
            metrics = SegmentationMetrics(num_classes)
            preds1 = preds.cpu().numpy()
            vids1 = vids.cpu().numpy()
            metrics.update_confusion_matrix(vids1, preds1)
            mean_iou = metrics.mean_iou()
            vloss = loss_fn(voutputs, vids)
            running_vloss += vloss
            viou = 1 - mean_iou
            running_viou += viou


    avg_vloss = running_vloss / (batch_idx + 1)
    avg_viou = running_viou / (batch_idx + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('IOU valid {}'.format(avg_viou))

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    
    writer.add_scalar('Validation IOU', avg_viou, epoch_number + 1)
    
    for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar('Learning Rate', current_lr, epoch_number + 1)

    writer.flush()


    scheduler.step(avg_viou)

    # Track best performance, and save the model's state
    if avg_viou < best_viou:
        best_viou = avg_viou
        model_path = config['save_model_path'] + '/mixedGT1_best_model_5'
        torch.save(model.state_dict(), model_path)
    if epoch_number == EPOCHS - 1:
        model_path = config['save_model_path'] + '/mixedGT1_last_model_5'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1