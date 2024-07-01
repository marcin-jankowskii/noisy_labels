import cv2

from models.Unet import UNet
from dataset.data import BatchMaker
from utils.metrics2 import calculate_iou, calculate_ap_for_segmentation
from utils.augmentation import MyAugmentation
from utils.metrics import SegmentationMetrics
from utils.Emilia_aug import EmiliaAugmentation
from utils.better_aug import BetterAugmentation
from utils.better_aug_2masks import BetterAugmentation as BetterAugmentation2

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
        elif len(data) == 6:
            inputs,intersections, unions,feelings,ids_y1, ids_y2 = data

        if T_aug == True:
            for i in range(inputs.shape[0]):
                if len(data) == 2:
                    inputs[i],ids[i] = augumentation(inputs[i], ids[i])
                elif len(data) == 6:
                    inputs[i],unions[i],intersections[i] = augumentation(inputs[i], unions[i],intersections[i])


        inputs = inputs.to(device)
        if len(data) == 2:
            if config.loss == 'CrossEntropyLoss':
                ids = ids.type(torch.LongTensor)
            if config.loss == 'BCEWithLogitsLoss':
                ids = ids.type(torch.FloatTensor)
            ids = ids.to(device)
            idsBCE = ids[:, [0, -1], :, :]
            idsBCE = idsBCE.to(device)
        elif len(data) == 6:

            intersections = intersections.type(torch.LongTensor)
            intersections = intersections.to(device)
            intersections1 = transform_batch(intersections.cpu())
            intersections2 = transform_batch2(intersections.cpu())
            intersections1 = torch.from_numpy(intersections1).type(torch.LongTensor).to(device)
            intersections2 = torch.from_numpy(intersections2).type(torch.LongTensor).to(device)

            ids_y1 =ids_y1.type(torch.LongTensor)
            ids_y1 = ids_y1.to(device)

            ids_y2 = ids_y2.type(torch.LongTensor)
            ids_y2 = ids_y2.to(device)

            ids = ids_y2.to(device)


            unions = unions.type(torch.LongTensor)
            unions = unions.to(device)
            unions1 = transform_batch(unions.cpu())
            unions2 = transform_batch2(unions.cpu())
            unions1 = torch.from_numpy(unions1).type(torch.LongTensor).to(device)
            unions2 = torch.from_numpy(unions2).type(torch.LongTensor).to(device)

            un_diff_inter = ids - intersections
            un_diff_inter = un_diff_inter.type(torch.LongTensor)
            un_diff_inter = un_diff_inter.to(device)

            un_diff_inter1 = transform_batch(un_diff_inter.cpu())
            un_diff_inter2 = transform_batch2(un_diff_inter.cpu())

            un_diff_inter1 = torch.from_numpy(un_diff_inter1).type(torch.LongTensor).to(device)
            un_diff_inter2 = torch.from_numpy(un_diff_inter2).type(torch.LongTensor).to(device)


        ids1 = transform_batch(ids.cpu())
        ids2 = transform_batch2(ids.cpu())

        # for i in range(ids1.shape[0]):
        #     ids1[i] = cv2.morphologyEx(ids1[i], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        ids1 = torch.from_numpy(ids1).type(torch.LongTensor).to(device)
        ids2 = torch.from_numpy(ids2).type(torch.LongTensor).to(device)



        # zamienic nazwy bo unions to intersections a intersections to y2
        optimizer.zero_grad()

        output = model(inputs)
        output1 = output[:, :3, :, :]
        output2 = output[:, [0, -1], :, :]

        weights1 = torch.tensor([0.2, 1.0, 0.5]).to(device)
        weights2 = torch.tensor([1.0, 1.0]).to(device)
        loss_fn1 = nn.CrossEntropyLoss(weight=weights1)
        loss_fn2 = nn.CrossEntropyLoss(weight=weights2)


        # Obliczanie straty
        
        if config.mode == 'intersection_and_union':
            k = config.k
            l1 = loss_fn1(output1, intersections1)  + k*loss_fn1(output1, un_diff_inter1)
            l2 = loss_fn2(output2, intersections2) + k*loss_fn2(output2, un_diff_inter2)
            loss = l1 + l2
            #loss = loss_fn(output, ids)
        elif config.mode == 'oneclass':
            if config.loss == 'CrossEntropyLoss':
                loss = loss_fn2(output, ids2)
            elif config.loss == 'BCEWithLogitsLoss':
                loss = loss_fn(output, idsBCE)
        elif config.mode == 'multiclass':
            if config.loss == 'CrossEntropyLoss':
                l1 = loss_fn1(output1, ids1)
                l2 = loss_fn2(output2, ids2)
                loss = l1 + l2
            if config.loss == 'BCEWithLogitsLoss':
                loss = loss_fn(output, ids)

        if config.mode == 'oneclass':
            preds2 = torch.argmax(output, dim=1)
            mean_iou, IoUs = calculate_iou(ids2.cpu().numpy(), preds2.cpu().numpy(),2)
            iou_oneclass = 1 - mean_iou

            loss.backward()
            optimizer.step()

            total_iou_multiclass += 0
            total_iou_oneclass += iou_oneclass
            total_loss += loss.item()

        elif config.mode == 'multiclass' or config.mode == 'intersection_and_union':
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
            if config.loss == 'CrossEntropyLoss':
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
    softmask_oneclass_list = []
    softmask_multiclass_list = []
    vids_list = []
    total_iou_multiclass = 0
    total_iou_oneclass = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):

            if len(data) == 2:
                vinputs, vids = data
                vids = vids.type(torch.FloatTensor)
                vids = vids.to(device)
            elif len(data) == 6:
                vinputs, vintersections, vunions, vfeelings, vids_y1, vids_y2 = data
                vintersections = vintersections.type(torch.FloatTensor)
                vids = vids_y1.to(device)
                vunions = vunions.type(torch.FloatTensor)
                
         
            vinputs = vinputs.to(device)
            images = vinputs.detach().cpu().numpy().transpose(0, 2, 3, 1)

            if config.mode == 'intersection_and_union':
                vids = vunions.to(device)

            if config.mode == 'oneclass':
                vids_list.append(vids.cpu())
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
                softmask_oneclass_list.append(vsofts2.cpu())



                #preds_out_multiclass = np.zeros((512, 512))
                preds_out_oneclass = vpreds2.detach().cpu().numpy()
                preds_out_multiclass = np.zeros_like(preds_out_oneclass)

                total_iou_multiclass = 0

                mean_iou, IoUs = calculate_iou(vids2.cpu().numpy(), vpreds2.cpu().numpy(),2)

                viou = 1 - mean_iou
                total_iou_oneclass += viou
            else:
                vids_list.append(vids.cpu())
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


                vsofts1 = torch.softmax(voutput1, dim=1)
                vsofts2 = torch.softmax(voutput2, dim=1)
                vsofts1 = vsofts1.squeeze(0)
                vsofts2 = vsofts2.squeeze(0)
                softmask_multiclass_list.append(vsofts1.cpu())
                softmask_oneclass_list.append(vsofts2.cpu())


                preds_out_multiclass = vpreds1.detach().cpu().numpy()
                preds_out_oneclass = vpreds2.detach().cpu().numpy()


                mean_iou, IoUs = calculate_iou(vids1.cpu().numpy(), vpreds1.cpu().numpy(),3)
                viou = 1 - mean_iou
                total_iou_multiclass += viou


                mean_iou, IoUs = calculate_iou(vids2.cpu().numpy(), vpreds2.cpu().numpy(),2)
                viou = 1 - mean_iou
                total_iou_oneclass += viou


    vids_list = np.concatenate(vids_list, axis=0)
    ids1 = transform_batch(vids_list)
    ids2 = transform_batch2(vids_list)


    softmasks_oneclass = [mask for batch in softmask_oneclass_list for mask in batch]
    softmasks_oneclass_np = np.array(softmasks_oneclass)
    softmasks_oneclass_np = softmasks_oneclass_np.transpose(0, 2, 3, 1)
    ap_score_oneclass = calculate_ap_for_segmentation(softmasks_oneclass_np[:, :, :, 1], ids2)
    ap_score_head = 0
    ap_score_tail = 0

    if config.mode == 'multiclass' or config.mode == 'intersection_and_union':
        softmasks_multiclass = [mask for batch in softmask_multiclass_list for mask in batch]
        softmasks_multiclass_np = np.array(softmasks_multiclass)
        softmasks_multiclass_np = softmasks_multiclass_np.transpose(0, 2, 3, 1)
        ap_score_head = calculate_ap_for_segmentation(softmasks_multiclass_np[:, :, :, 2], vids_list[:,2,:,:])
        ap_score_tail = calculate_ap_for_segmentation(softmasks_multiclass_np[:, :, :, 1], vids_list[:,1,:,:])


    avg_iou_multiclass = total_iou_multiclass / len(validation_loader)
    avg_iou_oneclass = total_iou_oneclass / len(validation_loader)


    if config.scheduler == 'ReduceLROnPlateau':
        scheduler.step(avg_iou_multiclass)

    val_metrics = { "val/val_iou_multiclass": avg_iou_multiclass,
                    "val/val_iou_oneclass": avg_iou_oneclass,
                    "val/val_ap_oneclass": ap_score_oneclass,
                    "val/val_ap_head": ap_score_head,
                    "val/val_ap_tail": ap_score_tail,
                    "val/epoch": epoch_number
                       }
    wandb.log(val_metrics)
    return avg_iou_multiclass,avg_iou_oneclass,images,vids_numpy,vids_numpy2,preds_out_multiclass,preds_out_oneclass,ap_score_oneclass,ap_score_head,ap_score_tail,

def main(model, train_loader, validation_loader, optimizer,scheduler,loss_fn, epochs,augumentation,T_aug,name):

    best_iou = 1000000
    best_iou_multiclass = 1000000
    best_iou_opt_oneclass = 0
    best_ap_oneclass = 0
    best_ap_head = 0
    best_ap_tail = 0

    for epoch in range(epochs):
        epoch_number = epoch +1
        train_loss,train_iou_multiclass,train_iou_oneclass = train(model, train_loader, optimizer,scheduler, loss_fn,augumentation,T_aug,epoch_number)
        validation_iou_multiclass,validation_iou_oneclass,vimages,vlbls_multiclass,vlbls_oneclass,vpreds_multiclass,vpreds_oneclass,ap_score_oneclass,ap_score_head,ap_score_tail = val(model, validation_loader, loss_fn,epoch_number,scheduler)
        plot_sample(vimages,vlbls_multiclass,vpreds_multiclass, ix=0,mode = 'val',number = '1')
        plot_sample(vimages, vlbls_oneclass, vpreds_oneclass, ix=0, mode='val',number = '2')

        print(f'Epoch {epoch_number}, Train Loss: {train_loss}, Train Iou Multiclass: {train_iou_multiclass}, Train Iou Oneclass: {train_iou_oneclass}, Validation Iou Multiclass: {validation_iou_multiclass}, Validation Iou Oneclass: {validation_iou_oneclass}')
        print(f'Validation AP Oneclass: {ap_score_oneclass}',f'Validation AP Head: {ap_score_head}',f'Validation AP Tail: {ap_score_tail}')

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
        if ap_score_head > best_ap_head:
            best_ap_head = ap_score_head
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_ap_head'
            torch.save(model.state_dict(), model_path)
            print("Model saved (ap_head)")
        if ap_score_tail > best_ap_tail:
            best_ap_tail = ap_score_tail
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_ap_tail'
            torch.save(model.state_dict(), model_path)
            print("Model saved (ap_tail)")
        if validation_iou_multiclass < best_iou_multiclass:
            best_iou_multiclass = validation_iou_multiclass
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_iou_multiclass'
            torch.save(model.state_dict(), model_path)
            print('Model saved (iou_multiclass)')
        if epoch_number == epochs:
            model_path = yaml_config['save_model_path'] +'/'+ name + '_last_model'
            torch.save(model.state_dict(), model_path)
            #wandb.save(model_path)
            print('Model saved')



class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]  # tło, wić, główka


timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


wandb.init(project="Noisy_label", entity="noisy_label",
            config={
            "epochs": 300,
            "batch_size": 6,
            "lr": 1e-3,
            "annotator": 1,
            "model": 'smpUNet++',
            "augmentation": False,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "place": "lab",
            "mode": "intersection_and_union",
            "aug_type": "BetterAugmentation",
            "k": 5

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
             'intersection_and_union': 'intersection_and_union_inference',
             'feeling_lucky': 'feeling_lucky',
             'union': 'union',
             "oneclass": 'mixed',
             "multiclass": 'mixed'
}

name = (f'Annotator:{config.annotator}_Model:{config.model}_Augmentation:{config.augmentation}_Mode:{config.mode}_Optimizer:{config.optimizer}_Scheduler:{config.scheduler}_Epochs:_{config.epochs}_Batch_Size:{config.batch_size}_Start_lr:{config.lr}_Loss:{config.loss}_Timestamp:{timestamp}')
save_name = name = (f'Annotator_{config.annotator}_Model_{config.model}_Augmentation_{config.augmentation}_Mode{config.mode}_Optimizer_{config.optimizer}_Scheduler_{config.scheduler}_Epochs_{config.epochs}_Batch_Size_{config.batch_size}_Start_lr_{config.lr}_Loss_{config.loss}_Timestamp_{timestamp}')

description = f'y2_Two_segment_loss alfa = {config.k}'
#description = f'One_segment_loss'
wandb.run.name = description + name




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

if config.mode == "multiclass" or config.mode == "intersection_and_union":
    weights = torch.ones([num_classes,512,512])
    weights[0] = 1
    weights[1] = 10 #7
    weights[2] = 7  #2
    weights[3] = 8  #4
    weights = weights.to(device)
if config.mode =="oneclass":
    weights = torch.ones([num_classes,512,512])
    weights[0] = 1
    weights[1] = 1
    weights = weights.to(device)

loss_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss(),
             'CrossEntropyLossWeight': nn.CrossEntropyLoss(),
             'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=weights),
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
    if config.mode == 'intersection_and_union':
        aug = BetterAugmentation2()
t_aug = config.augmentation
print(config.aug_type)
main(model, train_loader, val_loader, optimizer,scheduler, loss_fn, config.epochs,aug,t_aug,save_name)


