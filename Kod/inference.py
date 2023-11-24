from models.Unet import UNet
from dataset.data import BatchMaker
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import (
    jaccard_score,
    average_precision_score
)

BATCH = 1
#path_to_config = '/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
path_to_config = '/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml'
with open(path_to_config, 'r') as config_file:
    config = yaml.safe_load(config_file)
model_path = config['save_model_path'] + '/mixedGT1_best_model'


batch_maker = BatchMaker(config_path=path_to_config, batch_size=BATCH, mode = 'test',segment = 'mixed',annotator= 1)
test_loader = batch_maker.test_loader



def plot_sample(X, y, preds, binary_preds, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4,figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    #if has_mask:
        #ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Sperm Image')
    ax[0].set_axis_off()

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Sperm Mask Image')
    ax[1].set_axis_off()

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    #if has_mask:
        #ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Sperm Image Predicted')
    ax[2].set_axis_off()

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    #if has_mask:
        #ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Sperm Mask Image Predicted binary')
    ax[3].set_axis_off()
    plt.savefig(config['save_inf_fig_path']+'/{}.png'.format(ix))
    plt.close()


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")

model = UNet(3,1)
model.load_state_dict(torch.load(model_path)) 
model.to(device)
model.eval() 

# Listy do przechowywania obrazów wejściowych, predykcji i etykiet 
input_images = []
predicted_masks = []
true_masks = []  

# Pętla do przewidywania na danych testowych
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

    
        input_images.append(inputs.cpu())
        predicted_masks.append(outputs.cpu())
        true_masks.append(labels.cpu())  

input_images = np.concatenate(input_images, axis=0)
true_masks = np.concatenate(true_masks, axis=0) 
predicted_masks = torch.cat(predicted_masks, dim=0).cpu().numpy() 

# Threshold predictions
x_images = input_images.transpose((0, 2, 3, 1))
true = true_masks.reshape(119, 512, 512)
pred = predicted_masks.reshape(119, 512, 512)

threshold = 0.5
true_masks_t = (true > threshold).astype(np.uint8)
predicted_masks_t = (pred > threshold).astype(np.uint8)

for i in range(len(x_images)):
    plot_sample(x_images, true, pred, predicted_masks_t, ix=i)
    print('sample {} saved'.format(i))

#IoU = jaccard_score(true_masks_t.flatten(), predicted_masks_t.flatten())
#average_precision = average_precision_score(true_masks_t.flatten(), predicted_masks_t.flatten())

#print("IoU: {}".format(IoU))
#print("Average Precision: {}".format(average_precision))    