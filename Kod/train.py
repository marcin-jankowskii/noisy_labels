from models.Unet import UNet
from dataset.data import BatchMaker
from utils.metrics2 import calculate_iou, calculate_ap_for_segmentation
from utils.augmentation import MyAugmentation
from utils.metrics import SegmentationMetrics
from utils.Emilia_aug import EmiliaAugmentation
from utils.better_aug import BetterAugmentation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
import datetime
import yaml
import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
import segmentation_models_pytorch as smp


class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]

def transform_mask(mask):
    new_mask = np.zeros((512, 512))
    for i in range(3):
        new_mask[mask[i] == 1] = i

    return new_mask

def transform_mask2(mask):
    new_mask = np.zeros((512, 512))
    for i in range(4):
        if i == 0:
            new_mask[mask[i] == 1] = i
        if i == 3:
            new_mask[mask[i] == 1] = 1
    return new_mask

def transform_batch(batch):
    # Tworzenie nowego batcha o wymiarach (500, 512, 512)
    new_batch = np.zeros((batch.shape[0], 512, 512))

    # Przypisanie wartości 1, 2, 3 do odpowiednich warstw dla każdej maski w batchu
    for i in range(batch.shape[0]):
        new_batch[i] = transform_mask(batch[i])

    return new_batch

def transform_batch2(batch):
    # Tworzenie nowego batcha o wymiarach (500, 512, 512)
    new_batch = np.zeros((batch.shape[0], 512, 512))

    # Przypisanie wartości 1, 2, 3 do odpowiednich warstw dla każdej maski w batchu
    for i in range(batch.shape[0]):
        new_batch[i] = transform_mask2(batch[i])

    return new_batch

def plot_sample(X, y, preds, ix=None,mode = 'train', number = '1'):
    """Function to plot the results"""
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić
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
    ax[2].set_title('Sperm Image Predicted (class1 and class2)')
    ax[2].set_axis_off()




    if mode == 'train':
        wandb.log({"train/plot": wandb.Image(fig)})
    if mode == 'val':
        if number == '1':
            wandb.log({"val/plot": wandb.Image(fig)})
        else:
            wandb.log({"val/plot" + number: wandb.Image(fig)})

    plt.close()

def train(model, train_loader, optimizer,scheduler,loss_fn,augumentation,T_aug,epoch_number):
    model.train()
    total_loss = 0
    total_loss_oneclass = 0
    total_loss_multiclass = 0
    total_iou_multiclass = 0
    total_iou_oneclass = 0
    for batch_idx, data in enumerate(train_loader):
        if len(data) == 2:
            inputs, ids = data
        elif len(data) == 3: 
            inputs, intersections, unions = data

        if T_aug == True:
            for i in range(inputs.shape[0]):
                if len(data) == 2:
                    inputs[i],ids[i] = augumentation(inputs[i], ids[i])
                elif len(data) == 3:
                    inputs[i],intersections[i] = augumentation(inputs[i], intersections[i])


        inputs = inputs.to(device)
        if len(data) == 2:
            ids = ids.type(torch.LongTensor)
            ids = ids.to(device)
        elif len(data) == 3:
            intersections = intersections.type(torch.LongTensor)
            intersections = intersections.to(device)
            ids = intersections.to(device)
            unions = unions.type(torch.LongTensor)
            unions = unions.to(device)

            if config.mode == 'intersection_and_union':
                #inputs = intersections.to(device)
                unions = unions.type(torch.LongTensor)
                ids = unions.to(device)

        ids1 = transform_batch(ids.cpu())
        ids2 = transform_batch2(ids.cpu())

        ids1 = torch.from_numpy(ids1).type(torch.LongTensor).to(device)
        ids2 = torch.from_numpy(ids2).type(torch.LongTensor).to(device)

        # zamienic nazwy bo unions to intersections a intersections to y2
        optimizer.zero_grad()

        output = model(inputs)
        output1 = output[:, :3, :, :]
        output2 = output[:, [0, -1], :, :]

        weights1 = torch.tensor([0.2, 1, 0.5]).to(device)
        weights2 = torch.tensor([1.0, 1.0]).to(device)
        loss_fn1 = nn.CrossEntropyLoss(weight=weights1)
        loss_fn2 = nn.CrossEntropyLoss(weight=weights2)


        # Obliczanie straty
        
        if config.mode == 'intersection_and_union':
            loss = loss_fn(output, ids) + 2*loss_fn(output*unions, ids - unions)
            #loss = loss_fn(output, ids)
        elif config.mode == 'oneclass':
            loss = loss_fn2(output, ids2)
        else:
            l1 = loss_fn1(output1, ids1)
            l2 = loss_fn2(output2, ids2)
            loss = l1 + l2

        if config.mode == 'oneclass':
            preds2 = torch.argmax(output, dim=1)
            mean_iou, IoUs = calculate_iou(ids2.cpu().numpy(), preds2.cpu().numpy(),2)
            iou_oneclass = 1 - mean_iou

            loss.backward()
            optimizer.step()

            total_iou_multiclass += 0
            total_iou_oneclass += iou_oneclass
            total_loss += loss.item()
        else:
            preds1 = torch.argmax(output1, dim=1)
            preds2 = torch.argmax(output2, dim=1)


            mean_iou, IoUs = calculate_iou(ids1.cpu().numpy(), preds1.cpu().numpy(), 3)
            iou_multiclass = 1 - mean_iou
            mean_iou, IoUs = calculate_iou(ids2.cpu().numpy(), preds2.cpu().numpy(),2)
            iou_oneclass = 1 - mean_iou

            loss.backward()
            optimizer.step()

            total_iou_multiclass += iou_multiclass
            total_iou_oneclass += iou_oneclass
            total_loss += loss.item()
            total_loss_oneclass += l2.item()
            total_loss_multiclass += l1.item()
        
    avg_loss = total_loss / len(train_loader)
    avg_loss_oneclass = total_loss_oneclass / len(train_loader)
    avg_loss_multiclass = total_loss_multiclass / len(train_loader)
    avg_iou_multiclass = total_iou_multiclass / len(train_loader)
    avg_iou_oneclass = total_iou_oneclass / len(train_loader)


    if config.scheduler != 'ReduceLROnPlateau':
        scheduler.step()
    metrics = {"train/train_loss": avg_loss,
                "train/train_loss_oneclass": avg_loss_oneclass,
                "train/train_loss_multiclass": avg_loss_multiclass,
               "train/train_iou_multiclass": avg_iou_multiclass,
               "train/train_iou_oneclass": avg_iou_oneclass,
               "train/lr": optimizer.param_groups[0]['lr'],
                       "train/epoch": epoch_number
                       }
    


    wandb.log(metrics)

    return avg_loss,avg_iou_multiclass,avg_iou_oneclass

def val(model, validation_loader, loss_fn,epoch_number,scheduler):
    model.eval()
    total_iou_multiclass = 0
    total_iou_oneclass = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):

            if len(data) == 2:
                vinputs, vids = data
                vids = vids.type(torch.FloatTensor)
                vids = vids.to(device)
            elif len(data) == 3:
                vinputs, vintersections, vunions = data
                vintersections = vintersections.type(torch.FloatTensor)
                vids = vintersections.to(device)
                vunions = vunions.type(torch.FloatTensor)
                
         
            vinputs = vinputs.to(device)
            images = vinputs.detach().cpu().numpy().transpose(0, 2, 3, 1)

            if config.mode == 'intersection_and_union':
                vids = vunions.to(device)

            if config.mode == 'oneclass':
                vids1 = transform_batch(vids.cpu())
                vids2 = transform_batch2(vids.cpu())

                vids1 = torch.from_numpy(vids1).type(torch.LongTensor).to(device)
                vids2 = torch.from_numpy(vids2).type(torch.LongTensor).to(device)

                vids_numpy = vids1.detach().cpu().numpy()
                vids_numpy2 = vids2.detach().cpu().numpy()

                voutputs = model(vinputs)

                # voutputs_binary = torch.argmax(voutputs, dim=1)
                # preds_out = voutputs_binary.detach().cpu().numpy().transpose(0, 2, 3, 1)
                # meanIoU, IoUs = calculate_iou(vids.cpu().numpy(), voutputs_binary.detach().cpu().numpy())

                vpreds2 = torch.argmax(voutputs, dim=1)
                vsofts2 = torch.softmax(voutputs, dim=1)
                vsofts2 = vsofts2.squeeze(0)
                vsoftmask_oneclass_np = np.array(vsofts2.cpu())
                vsoftmask_oneclass_np = vsoftmask_oneclass_np.transpose((0, 2, 3, 1))


                #preds_out_multiclass = np.zeros((512, 512))
                preds_out_oneclass = vpreds2.detach().cpu().numpy()
                preds_out_multiclass = np.zeros_like(preds_out_oneclass)

                total_iou_multiclass = 0

                mean_iou, IoUs = calculate_iou(vids2.cpu().numpy(), vpreds2.cpu().numpy(),2)
                ap_score_oneclass = calculate_ap_for_segmentation(vsoftmask_oneclass_np[:, :, :, 1], vids2.cpu().numpy())
                viou = 1 - mean_iou
                total_iou_oneclass += viou
            else:
                vids1 = transform_batch(vids.cpu())
                vids2 = transform_batch2(vids.cpu())

                vids1 = torch.from_numpy(vids1).type(torch.LongTensor).to(device)
                vids2 = torch.from_numpy(vids2).type(torch.LongTensor).to(device)

                vids_numpy = vids1.detach().cpu().numpy()
                vids_numpy2 = vids2.detach().cpu().numpy()

                voutputs = model(vinputs)
                voutput1 = voutputs[:, :3, :, :]
                voutput2 = voutputs[:, [0, -1], :, :]


                # voutputs_binary = torch.argmax(voutputs, dim=1)
                # preds_out = voutputs_binary.detach().cpu().numpy().transpose(0, 2, 3, 1)
                # meanIoU, IoUs = calculate_iou(vids.cpu().numpy(), voutputs_binary.detach().cpu().numpy())

                vpreds1 = torch.argmax(voutput1, dim=1)
                vpreds2 = torch.argmax(voutput2, dim=1)

                preds_out_multiclass = vpreds1.detach().cpu().numpy()
                preds_out_oneclass = vpreds2.detach().cpu().numpy()


                mean_iou, IoUs = calculate_iou(vids1.cpu().numpy(), vpreds1.cpu().numpy(),3)
                viou = 1 - mean_iou
                total_iou_multiclass += viou


                mean_iou, IoUs = calculate_iou(vids2.cpu().numpy(), vpreds2.cpu().numpy(),2)
                viou = 1 - mean_iou
                total_iou_oneclass += viou




    avg_iou_multiclass = total_iou_multiclass / len(validation_loader)
    avg_iou_oneclass = total_iou_oneclass / len(validation_loader)

    if config.scheduler == 'ReduceLROnPlateau':
        scheduler.step(avg_iou_multiclass)

    val_metrics = { "val/val_iou_multiclass": avg_iou_multiclass,
                    "val/val_iou_oneclass": avg_iou_oneclass,
                    "val/val_ap_oneclass": ap_score_oneclass,
                    "val/epoch": epoch_number
                       }
    wandb.log(val_metrics)
    return avg_iou_multiclass,avg_iou_oneclass,images,vids_numpy,vids_numpy2,preds_out_multiclass,preds_out_oneclass,ap_score_oneclass

def main(model, train_loader, validation_loader, optimizer,scheduler,loss_fn, epochs,augumentation,T_aug,name):

    best_iou = 1000000
    best_ap_oneclass = 0

    for epoch in range(epochs):
        epoch_number = epoch +1
        train_loss,train_iou_multiclass,train_iou_oneclass = train(model, train_loader, optimizer,scheduler, loss_fn,augumentation,T_aug,epoch_number)
        validation_iou_multiclass,validation_iou_oneclass,vimages,vlbls_multiclass,vlbls_oneclass,vpreds_multiclass,vpreds_oneclass,ap_score_oneclass = val(model, validation_loader, loss_fn,epoch_number,scheduler)
        plot_sample(vimages,vlbls_multiclass,vpreds_multiclass, ix=0,mode = 'val',number = '1')
        plot_sample(vimages, vlbls_oneclass, vpreds_oneclass, ix=0, mode='val',number = '2')

        print(f'Epoch {epoch_number}, Train Loss: {train_loss}, Train Iou Multiclass: {train_iou_multiclass}, Train Iou Oneclass: {train_iou_oneclass}, Validation Iou Multiclass: {validation_iou_multiclass}, Validation Iou Oneclass: {validation_iou_oneclass}')

        if validation_iou_oneclass < best_iou:
            best_iou = validation_iou_oneclass
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_iou_oneclass'
            torch.save(model.state_dict(), model_path)
            #wandb.save(model_path)
            print('Model saved (iou_oneclass)')
        if ap_score_oneclass > best_ap_oneclass:
            best_ap_oneclass = ap_score_oneclass
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_ap_oneclass'
            torch.save(model.state_dict(), model_path)
            print("Model saved (ap_oneclass)")
        if epoch_number == epochs:
            model_path = yaml_config['save_model_path'] +'/'+ name + '_last_model'
            torch.save(model.state_dict(), model_path)
            #wandb.save(model_path)
            print('Model saved')



class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]  # tło, wić, główka


timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")



wandb.init(project="noisy_labels", entity="segsperm",
            config={
            "epochs": 200,
            "batch_size": 6,
            "lr": 1e-3,
            "annotator": 1,
            "model": 'smpUNet++',
            "augmentation": True,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "scheduler": "MultiStepLR",
            "place": "lab",
            "mode": "oneclass",
            "aug_type": "EmiliaAugmentation"
            })

config = wandb.config


if config.mode == 'oneclass':
    num_classes = 2
else:
    num_classes = 4


path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            }

model_dict = {'myUNet': UNet(3,num_classes),
              'smpUNet': smp.Unet(in_channels = 3, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels = 3, classes=num_classes,encoder_name="resnet18",encoder_weights=None),
              'MAnet': smp.MAnet(in_channels = 3, classes=num_classes,encoder_name="resnet18",encoder_weights=None),
              'DeepLabV3+': smp.DeepLabV3Plus(in_channels = 3, classes=num_classes,encoder_name="resnet18",encoder_weights=None)
}

mode_dict = {'normal': 'mixed',
             'intersection': 'intersection',
             'intersection_and_union': 'intersection_and_union',
             'feeling_lucky': 'feeling_lucky',
             'union': 'union',
             "oneclass": 'mixed',
             "multiclass": 'mixed'
}

name = (f'Annotator:{config.annotator}_Model:{config.model}_Augmentation:{config.augmentation}_Mode:{config.mode}_Optimizer:{config.optimizer}_Scheduler:{config.scheduler}_Epochs:_{config.epochs}_Batch_Size:{config.batch_size}_Start_lr:{config.lr}_Loss:{config.loss}_Timestamp:{timestamp}')
save_name = name = (f'Annotator_{config.annotator}_Model_{config.model}_Augmentation_{config.augmentation}_Mode{config.mode}_Optimizer_{config.optimizer}_Scheduler_{config.scheduler}_Epochs_{config.epochs}_Batch_Size_{config.batch_size}_Start_lr_{config.lr}_Loss_{config.loss}_Timestamp_{timestamp}')
wandb.run.name = name




with open(path_dict[config.place], 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

# batch_maker modes: train, val, test
# batch_maker segments: full, head, tail, mixed, intersection_and_union
batch_maker = BatchMaker(config_path=path_dict[config.place], batch_size=config.batch_size,mode ='train',segment = mode_dict[config.mode],annotator= config.annotator)
train_loader = batch_maker.train_loader
val_loader = batch_maker.test_loader


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

loss_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss(),
             'CrossEntropyLossWeight': nn.CrossEntropyLoss(),
             'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
             'BCE': nn.BCELoss()}

loss_fn = loss_dict[wandb.config.loss]

optimizer = optimizer_dict[config.optimizer]

scheduler_dict = {'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=config.epochs),
                  'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min'),
                  "MultiStepLR": MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3),
                  'None': None}

scheduler = scheduler_dict[config.scheduler]
wandb.watch(model, log="all")
if config.aug_type == 'EmiliaAugmentation':
    aug = EmiliaAugmentation()
elif config.aug_type == 'MyAugmentation':
    aug = MyAugmentation()
elif config.aug_type == 'BetterAugmentation':
    aug = BetterAugmentation()
t_aug = config.augmentation
main(model, train_loader, val_loader, optimizer,scheduler, loss_fn, config.epochs,aug,t_aug,save_name)


