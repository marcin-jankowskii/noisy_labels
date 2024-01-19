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
import segmentation_models_pytorch as smp


def plot_sample(X, y, preds, ix=None,mode = 'train'):
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
    if mode == 'train':
        wandb.log({"train/plot": wandb.Image(fig)})
    if mode == 'val':
        wandb.log({"val/plot": wandb.Image(fig)})
    plt.close()

def train(model, train_loader, optimizer,scheduler,loss_fn,augumentation,T_aug,epoch_number):
    model.train()
    total_loss = 0
    total_iou = 0
    for batch_idx, (inputs,ids) in enumerate(train_loader):
        
        if T_aug == True:
            for i in range(inputs.shape[0]):
                inputs[i],ids[i] = augumentation(inputs[i], ids[i])

        ids = ids.type(torch.LongTensor)
        inputs = inputs.to(device)
        ids = ids.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        images = inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        ids_numpy = ids.detach().cpu().numpy()

        preds_out = output.detach().cpu().numpy()
        preds_out = np.argmax(preds_out, axis=1)

        preds = torch.argmax(output, dim=1) 
        metrics = SegmentationMetrics(num_classes)
        metrics.update_confusion_matrix(ids.cpu().numpy(), preds.cpu().numpy())
        mean_iou = metrics.mean_iou()
        viou = 1 - mean_iou

        loss = loss_fn(output, ids)
        loss.backward()
        optimizer.step()

        total_iou += viou
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)


    if config.scheduler != 'ReduceLROnPlateau':
        scheduler.step()
    metrics = {"train/train_loss": avg_loss, 
               "train/train_iou": avg_iou,
               "train/lr": optimizer.param_groups[0]['lr'],
                       "train/epoch": epoch_number
                       }
    
 
    wandb.log(metrics)

    return avg_loss,avg_iou,images,ids_numpy,preds_out

def val(model, validation_loader, loss_fn,epoch_number,scheduler):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for batch_idx, (vinputs,vids) in enumerate(val_loader):
            vids = vids.type(torch.LongTensor)
            vinputs = vinputs.to(device)
            images = vinputs.cpu().numpy().transpose(0, 2, 3, 1)
            vids = vids.to(device)
            vids_numpy = vids.cpu().numpy()
            voutputs = model(vinputs)
            preds = torch.argmax(voutputs, dim=1) 
            preds_out = preds.cpu().numpy()
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
    return avg_loss,avg_iou,images,vids_numpy,preds_out

def main(model, train_loader, validation_loader, optimizer,scheduler,loss_fn, epochs,augumentation,T_aug,name):

    best_iou = 1000000

    for epoch in range(epochs):
        epoch_number = epoch +1
        train_loss,train_iou,images,lbls,preds = train(model, train_loader, optimizer,scheduler, loss_fn,augumentation,T_aug,epoch_number)
        validation_loss, validation_iou,vimages,vlbls,vpreds = val(model, validation_loader, loss_fn,epoch_number,scheduler)
        plot_sample(images, lbls,preds, ix=0,mode = 'train')
        plot_sample(vimages,vlbls,vpreds, ix=0,mode = 'val')

        print(f'Epoch {epoch_number}, Train Loss: {train_loss}, Train Iou: {train_iou}, Validation Loss: {validation_loss}, Validation IOU: {validation_iou}')

        if validation_iou < best_iou:
            best_iou = validation_iou
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model'
            torch.save(model.state_dict(), model_path)
            #wandb.save(model_path)
            print('Model saved')
        if epoch_number == epochs:
            model_path = yaml_config['save_model_path'] +'/'+ name + '_last_model'
            torch.save(model.state_dict(), model_path)
            #wandb.save(model_path)
            print('Model saved')

num_classes = 3



path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            }

model_dict = {'myUNet': UNet(3,num_classes),
              'smpUNet': smp.Unet(in_channels = 3, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels = 3, classes=num_classes),
}


timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


wandb.init(project="noisy_labels", entity="segsperm",
            config={
            "epochs": 300,
            "batch_size": 22,
            "lr": 1e-4,
            "annotator": 1,
            "model": 'smpUNet++',
            "augmentation": True,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "place": "lab"
            })

config = wandb.config

name = (f'Annotator:{config.annotator}_Model:{config.model}_Augmentation:{config.augmentation}_Optimizer:{config.optimizer}_Scheduler:{config.scheduler}_Epochs:_{config.epochs}_Batch_Size:{config.batch_size}_Start_lr:{config.lr}_Loss:{config.loss}_Timestamp:{timestamp}')
save_name = name = (f'Annotator_{config.annotator}_Model_{config.model}_Augmentation_{config.augmentation}_Optimizer_{config.optimizer}_Scheduler_{config.scheduler}_Epochs_{config.epochs}_Batch_Size_{config.batch_size}_Start_lr_{config.lr}_Loss_{config.loss}_Timestamp_{timestamp}')
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


model = model_dict[config.model]
model.to(device)

optimizer_dict = {'Adam': optim.Adam(model.parameters(), lr=config.lr),
                  'SGD': optim.SGD(model.parameters(), lr=config.lr),
                  'RMSprop': optim.RMSprop(model.parameters(), lr=config.lr)
                  }

weights = torch.ones(num_classes)
weights[0] = 0.1
weights[1] = 0.7
weights[2] = 0.4
weights = weights.to(device)

loss_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss(),
             'CrossEntropyLossWeight': nn.CrossEntropyLoss(weight=weights)}

loss_fn = loss_dict[wandb.config.loss]

optimizer = optimizer_dict[config.optimizer]

scheduler_dict = {'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=config.epochs),
                  'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min'),
                  'None': None}

scheduler = scheduler_dict[config.scheduler]
wandb.watch(model, log="all")
aug = MyAugmentation()
t_aug = config.augmentation
main(model, train_loader, val_loader, optimizer,scheduler, loss_fn, config.epochs,aug,t_aug,save_name)


