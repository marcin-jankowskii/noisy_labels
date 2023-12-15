from models.Unet import UNet
from dataset.data import BatchMaker
from utils.metrics import SegmentationMetrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def train(model, train_loader, optimizer,scheduler,loss_fn,augumentation,T_aug = False):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels,ids) in enumerate(train_loader):

        if T_aug == True:
            for i in range(inputs.shape[0]):
                inputs[i], labels[i] = augumentation(inputs[i], labels[i])


        inputs = inputs.to(device)
        labels = labels.to(device)
        ids = ids.to(device)


        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step()

    #Tutaj napisz zapisywanie do wandb

    return avg_loss

def val(model, validation_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for batch_idx, (vinputs, vlabels,vids) in enumerate(val_loader):
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            vids = vids.to(device)
            voutputs = model(vinputs)
            preds = torch.argmax(voutputs, dim=1) 
         

            metrics = SegmentationMetrics(num_classes)
            metrics.update_confusion_matrix(vids.cpu().numpy(), preds.cpu().numpy())
            mean_iou = metrics.mean_iou()
            viou = 1 - mean_iou
            total_iou += viou
            loss = loss_fn(voutputs, vids)
            total_loss += loss.item()
    avg_loss = total_loss / len(validation_loader)
    avg_iou = total_iou / len(validation_loader)

    #Dodaj wysyłanie wandb
    return avg_loss,avg_iou

def main(model, train_loader, validation_loader, optimizer, loss_fn, epochs,augumentation,T_aug):
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer,scheduler, loss_fn,augumentation,T_aug)
        validation_loss, validation_iou = val(model, validation_loader, loss_fn)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Validation IOU: {validation_iou}')

# Przykładowe wywołanie
# main_loop(your_model, train_loader, validation_loader, optimizer, loss_function, num_epochs)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")

model = UNet(3,num_classes)
model.to(device)
weights = torch.ones(num_classes)
# weights[0] = 0.1
# weights[1] = 0.7
# weights[2] = 0.4
weights = weights.to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
aug = MyAugmentation()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
wandb.init(project="noisy_labels", entity="segsperm")
wandb.run.name = writer
wandb.watch(model, log="all")

main(model, train_loader, val_loader, optimizer,scheduler, loss_fn, EPOCHS,aug,T_aug = False)


