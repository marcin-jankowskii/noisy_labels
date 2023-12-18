from models.Unet import UNet
from dataset.data import BatchMaker
from utils.metrics import SegmentationMetrics
from utils.augmentation import MyAugmentation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import datetime
import yaml
import matplotlib.pyplot as plt
import numpy as np
import wandb
import random

def plot_sample(X, y, preds, ix=None):
    """Function to plot the results"""
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 3,figsize=(20, 10))
    ax[0].imshow(X[ix])
    #if has_mask:
        #ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Sperm Image')
    ax[0].set_axis_off()


    mask_to_display = y[ix]
    print(mask_to_display.shape)

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask_to_display == i] = color


    ax[1].imshow(mask_rgb)
    ax[1].set_title('Sperm Mask Image')
    ax[1].set_axis_off()



    mask_to_display = preds[ix]

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask_to_display == i] = color
 

    ax[2].imshow(mask_rgb)
    #if has_mask:
        #ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Sperm Image Predicted')
    ax[2].set_axis_off()
    wandb.log({"train/plot": wandb.Image(fig)})
    plt.close()

def train(model, train_loader, optimizer,scheduler,loss_fn,augumentation,T_aug,epoch_number):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels,ids) in enumerate(train_loader):

        if T_aug == True:
            for i in range(inputs.shape[0]):
                inputs[i] = augumentation(inputs[i], labels[i])


        inputs = inputs.to(device)
        labels = labels.to(device)
        ids = ids.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        images = inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        lbls = labels.detach().cpu().numpy()
        preds = output.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

        loss = loss_fn(output, ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    if config.scheduler != 'ReduceLROnPlateau':
        scheduler.step()
    metrics = {"train/train_loss": avg_loss, 
               "train/lr": optimizer.param_groups[0]['lr'],
                       "train/epoch": epoch_number
                       }
    
 
    wandb.log(metrics)

    return avg_loss,images,lbls,preds

def val(model, validation_loader, loss_fn,epoch_number,scheduler):
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

    if config.scheduler == 'ReduceLROnPlateau':
        scheduler.step(avg_iou)

    val_metrics = {"val/val_loss": avg_loss, 
                    "val/val_iou": avg_iou,
                    "val/epoch": epoch_number
                       }
    wandb.log(val_metrics)
    return avg_loss,avg_iou

def main(model, train_loader, validation_loader, optimizer,scheduler,loss_fn, epochs,augumentation,T_aug,name):

    best_iou = 1000000

    for epoch in range(epochs):
        epoch_number = epoch +1
        train_loss,images,lbls,preds = train(model, train_loader, optimizer,scheduler, loss_fn,augumentation,T_aug,epoch_number)
        validation_loss, validation_iou = val(model, validation_loader, loss_fn,epoch_number,scheduler)
        plot_sample(images, lbls,preds, ix=0)

        print(f'Epoch {epoch_number}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Validation IOU: {validation_iou}')

        if validation_iou < best_iou:
            best_iou = validation_iou
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model'
            torch.save(model.state_dict(), model_path)
            #wandb.save(model_path)
            print('Model saved')

num_classes = 3
weights = torch.ones(num_classes)
weights[0] = 0.1
weights[1] = 0.7
weights[2] = 0.4


loss_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss(),
             'CrossEntropyLossWeight': nn.CrossEntropyLoss(weight=weights)}

path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            }


timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


wandb.init(project="noisy_labels", entity="segsperm",
            config={
            "epochs": 1,
            "batch_size": 3,
            "lr": 1e-4,
            "annotator": 1,
            "augmentation": True,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "place": "laptop"
            })

config = wandb.config

name = (f'Annotator:{config.annotator}, Augmentation:{config.augmentation}, Optimizer:{config.optimizer}, Scheduler:{config.scheduler}, Epochs: {config.epochs},Batch_Size:{config.batch_size}, Start_lr:{config.lr}, Loss: {config.loss}, Timestamp: {timestamp}')
wandb.run.name = name


with open(path_dict[config.place], 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

batch_maker = BatchMaker(config_path=path_dict[config.place], batch_size=config.batch_size,mode ='train',segment = 'mixed',annotator= config.annotator)
train_loader = batch_maker.train_loader
val_loader = batch_maker.val_loader


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")


model = UNet(3,num_classes)
model.to(device)

optimizer_dict = {'Adam': optim.Adam(model.parameters(), lr=config.lr),
                  'SGD': optim.SGD(model.parameters(), lr=config.lr),
                  'RMSprop': optim.RMSprop(model.parameters(), lr=config.lr)
                  }

weights = weights.to(device)
loss_fn = loss_dict[wandb.config.loss]

optimizer = optimizer_dict[config.optimizer]

scheduler_dict = {'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=config.epochs),
                  'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min'),
                  'None': None}

scheduler = scheduler_dict[config.scheduler]
wandb.watch(model, log="all")
aug = MyAugmentation()
t_aug = config.augmentation
main(model, train_loader, val_loader, optimizer,scheduler, loss_fn, config.epochs,aug,t_aug,name)


