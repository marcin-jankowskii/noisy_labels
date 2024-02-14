from models.Unet import UNet
from dataset.data import BatchMaker
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
import datetime
from utils.metrics import SegmentationMetrics
import segmentation_models_pytorch as smp

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
    plt.savefig(yaml_config['save_inf_fig_path']+'/{}.png'.format(ix))
    wandb.log({"inference/plot": wandb.Image(fig)})
    plt.close()

def predict(model, test_loader):
    model.eval() 
    input_images = []
    predicted_masks = []
    true_masks = []  
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

        
            input_images.append(inputs.cpu())
            predicted_masks.append(outputs.cpu())
            true_masks.append(ids.cpu())  

    input_images = np.concatenate(input_images, axis=0)
    true_masks = np.concatenate(true_masks, axis=0) 
    predicted_masks = np.concatenate(predicted_masks, axis=0)

    x_images = input_images.transpose((0, 2, 3, 1))
    true = true_masks
    pred = np.argmax(predicted_masks, axis=1)


    for i in range(len(x_images)):
        plot_sample(x_images, true,pred, ix=i)
        print('sample {} saved'.format(i))

    metrics = SegmentationMetrics(num_classes)
    metrics.update_confusion_matrix(true, pred)
    mean_iou = metrics.mean_iou()
    iou = metrics.calculate_iou_per_class()
    print("Mean IoU:", mean_iou)
    print("IoU dla każdej klasy:", iou)

    test_metrics =  {"inference/Mean Iou": mean_iou, 
                     "inference/Iou for each class": iou,
                       }
    wandb.log(test_metrics)


num_classes = 3
path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            } 

model_dict = {'myUNet': UNet(3,num_classes),
              'smpUNet': smp.Unet(in_channels = 3, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels = 3, classes=num_classes),
}   

wandb.init(project="noisy_labels", entity="segsperm",
            config={
            "model": "smpUNet++",
            "batch_size": 1,
            "annotator": 1,
            "place": 'lab'
            })

config = wandb.config

with open(path_dict[config.place], 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

saved_model_name = 'Annotator_1_Model_smpUNet++_Augmentation_False_Optimizer_Adam_Scheduler_CosineAnnealingLR_Epochs_300_Batch_Size_22_Start_lr_0.0001_Loss_CrossEntropyLossWeight_Timestamp_2024-02-09-11-26_best_model'    
model_path = yaml_config['save_model_path'] + '/' + saved_model_name
name = (f'Inference: Model_name: {saved_model_name}')

wandb.run.name = name
batch_maker = BatchMaker(config_path=path_dict[config.place], batch_size=config.batch_size, mode = 'test',segment = 'mixed',annotator= config.annotator)
test_loader = batch_maker.test_loader
num_classes = 3

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")

model = model_dict[config.model]
model.load_state_dict(torch.load(model_path)) 
model.to(device)
predict(model, test_loader)