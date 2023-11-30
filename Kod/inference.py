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

num_classes = 3
BATCH = 1
#path_to_config = '/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
#path_to_config = '/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml'
path_to_config = '/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml'
with open(path_to_config, 'r') as config_file:
    config = yaml.safe_load(config_file)
model_path = config['save_model_path'] + '/mixedGT1_best_model_4'


batch_maker = BatchMaker(config_path=path_to_config, batch_size=BATCH, mode = 'test',segment = 'mixed',annotator= 2)
test_loader = batch_maker.test_loader



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
    mask_to_display = np.argmax(mask_to_display, axis=0)

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask_to_display == i] = color


    ax[1].imshow(mask_rgb)
    ax[1].set_title('Sperm Mask Image')
    ax[1].set_axis_off()



    mask_to_display = preds[ix]
    mask_to_display = np.argmax(mask_to_display, axis=0)

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask_to_display == i] = color
 

    ax[2].imshow(mask_rgb)
    #if has_mask:
        #ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Sperm Image Predicted')
    ax[2].set_axis_off()
    plt.savefig(config['save_inf_fig_path']+'/{}.png'.format(ix))
    plt.close()


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")

model = UNet(3,num_classes)
model.load_state_dict(torch.load(model_path)) 
model.to(device)
model.eval() 

# Listy do przechowywania obrazów wejściowych, predykcji i etykiet 
input_images = []
predicted_masks = []
true_masks = []  

# Pętla do przewidywania na danych testowych
with torch.no_grad():
    for inputs, labels, ids in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

    
        input_images.append(inputs.cpu())
        predicted_masks.append(outputs.cpu())
        true_masks.append(labels.cpu())  

input_images = np.concatenate(input_images, axis=0)
true_masks = np.concatenate(true_masks, axis=0) 
#predicted_masks = torch.cat(predicted_masks, dim=0).cpu().numpy() 
predicted_masks = np.concatenate(predicted_masks, axis=0) 

# Threshold predictions
x_images = input_images.transpose((0, 2, 3, 1))
true = true_masks#.transpose((0, 2, 3, 1))
pred = predicted_masks#.transpose((0, 2, 3, 1))



for i in range(len(x_images)):
    plot_sample(x_images, true, pred, ix=i)
    print('sample {} saved'.format(i))

all_true_class_ids = []
all_pred_class_ids = []    

for true, pred in zip(true, pred):
    # Przekształć maski z formatu one-hot do formatu identyfikatorów klas
    true_class_id = np.argmax(true, axis=0)
    pred_class_id = np.argmax(pred, axis=0)
    # Dodaj identyfikatory klas do list
    all_true_class_ids.append(true_class_id.flatten())
    all_pred_class_ids.append(pred_class_id.flatten())

# Połącz wszystkie identyfikatory klas w jedną listę
all_true_class_ids = np.concatenate(all_true_class_ids)
all_pred_class_ids = np.concatenate(all_pred_class_ids)

IoU_per_class = jaccard_score(all_true_class_ids, all_pred_class_ids, average=None)

for i, IoU in enumerate(IoU_per_class):
    print(f'Jaccard score for class {i}: {IoU}')


