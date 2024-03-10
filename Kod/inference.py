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
from utils.metrics2 import calculate_iou
import segmentation_models_pytorch as smp
import cv2

def plot_sample(X, y, preds, ix=None, number = '1'):

    """Function to plot the results"""
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić

    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 5,figsize=(20, 10))
    ax[0].imshow(X[ix])
    #if has_mask:
        #ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Sperm Image')
    ax[0].set_axis_off()


    mask_to_display = y[ix]

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color


    ax[1].imshow(mask_rgb)
    ax[1].set_title('Sperm Mask Image')
    ax[1].set_axis_off()

    mask_to_display = preds[ix]

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color
 

    ax[2].imshow(mask_rgb)
    #if has_mask:
        #ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Sperm Image Predicted')
    ax[2].set_axis_off()


    # Utwórz obraz RGB z maski
    mask_rgb2 = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colorsV2):
        if class_id == 0 or class_id == 3:
            mask_rgb2[mask_to_display[:, :, class_id] == 1] = color


    ax[3].imshow(mask_rgb2)
    ax[3].set_title('Sperm Image Predicted (class3)')
    ax[3].set_axis_off()


    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        if class_id == 1 or class_id == 2:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = [0,255,255]
        else:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    ax[4].imshow(mask_rgb-mask_rgb2)
    ax[4].set_title('Diff (class1,2 - class3)')
    ax[4].set_axis_off()

    
    plt.savefig(yaml_config['save_inf_fig_path']+'/{}.png'.format(ix))
    if number == '1':
        wandb.log({"inference/plot": wandb.Image(fig)})
    else:
        wandb.log({"inference/plot"+number: wandb.Image(fig)})
    plt.close()

def video_sample(true,pred,ix=None):
    if ix is None:
        ix = random.randint(0, len(true))

    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić

    
    # mask - pred
    mask_to_display = true[ix]

    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    mask_to_display = pred[ix]

    mask_rgb2 = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb2[mask_to_display[:, :, class_id] == 1] = color

    union = cv2.bitwise_or(mask_rgb, mask_rgb2)
    diff = (union - mask_rgb2).transpose((2, 0, 1))


    # Róznica pomiędzy klasami 1,2 a 3
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        if class_id == 1 or class_id == 2:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = [0,255,255]
        else:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    mask_rgb2 = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colorsV2):
        if class_id == 0 or class_id == 3:
            mask_rgb2[mask_to_display[:, :, class_id] == 1] = color

    union = cv2.bitwise_or(mask_rgb, mask_rgb2)
    diff2 = (union - mask_rgb2).transpose((2, 0, 1))


    
    
    return diff,diff2
    
def video_sample2(intersection,union,pred,ix=None):
    if ix is None:
        ix = random.randint(0, len(intersection))

    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić

    ## (Union - Intersection) - (pred - Intersection)
    mask_to_display = intersection[ix]

    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    mask_to_display2 = union[ix]

    mask_rgb2 = np.zeros((mask_to_display2.shape[0], mask_to_display2.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb2[mask_to_display2[:, :, class_id] == 1] = color


    mask_to_display3 = pred[ix]

    mask_rgb3 = np.zeros((mask_to_display3.shape[0], mask_to_display3.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb3[mask_to_display3[:, :, class_id] == 1] = color

    union1 = cv2.bitwise_or(mask_rgb, mask_rgb2)
    union2 = cv2.bitwise_or(mask_rgb3, mask_rgb)
    udiff = union1 - mask_rgb
    pdiff = union2 - mask_rgb
    union3 = cv2.bitwise_or(udiff, pdiff)
    diff = (union3 - pdiff).transpose((2, 0, 1))

    return diff

def predict(model, test_loader):
    model.eval() 
    input_images = []
    predicted_masks = []
    true_masks = []
    true_masks2 = []
    unions_list =[]
    intersections_list = []
    with torch.no_grad():
        for data in test_loader:
            if len(data) == 2:
                inputs, ids = data
            elif len(data) == 3:
                if config.mode == 'intersection_and_union' or config.mode == 'intersection': 
                    inputs, intersections, unions = data
                else:
                    inputs, ids1, ids2 = data
            elif len(data) == 5:
                inputs,intersections, unions, ids1, ids2 = data

            inputs = inputs.to(device)
            outputs = model(inputs)

            if len(data) == 2:
                true_masks.append(ids.cpu())  
            elif len(data) == 3:
                if config.mode == 'intersection_and_union':
                    true_masks.append(unions.cpu())  
                    true_masks2.append(intersections.cpu())
                elif config.mode == 'intersection':
                    true_masks.append(intersections.cpu())
                elif config.mode == 'both':
                    true_masks.append(ids1.cpu())
                    true_masks2.append(ids2.cpu())
            elif len(data) == 5:
                if config.mode == 'intersection_and_union_inference' or config.mode == 'intersection_inference':
                    true_masks.append(ids1.cpu())  
                    true_masks2.append(ids2.cpu())
                    unions_list.append(unions.cpu())
                    intersections_list.append(intersections.cpu())


            input_images.append(inputs.cpu())
            outputs_binary = (outputs > 0.5).type(torch.FloatTensor) 
            predicted_masks.append(outputs_binary.cpu())

    input_images = np.concatenate(input_images, axis=0)
    true_masks = np.concatenate(true_masks, axis=0)
    predicted_masks = np.concatenate(predicted_masks, axis=0)

    x_images = input_images.transpose((0, 2, 3, 1))
    true = true_masks.transpose((0, 2, 3, 1))
    pred = predicted_masks.transpose((0, 2, 3, 1))

    if config.mode == 'intersection_and_union':
        true_masks2 = np.concatenate(true_masks2, axis=0)
        intersection = true_masks2.transpose((0, 2, 3, 1))

    
    if config.mode == 'both':

        mean_iou, iou = calculate_iou(true_masks, predicted_masks)
    
        diff = []
        diff3 = []

        for i in range(len(x_images)):
            plot_sample(x_images, true,pred, ix=i)
            df1,df3 = video_sample(true,pred,ix=i)
            diff.append(df1)
            diff3.append(df3)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        diff3 = np.array(diff3)    
        wandb.log({"video": wandb.Video(diff,fps=1)})
        wandb.log({"classVideo": wandb.Video(diff3,fps=1)})
        print("Mean IoU (annotator1):", mean_iou)
        print("IoU dla każdej klasy(annotator1):", iou)


        true_masks2 = np.concatenate(true_masks2, axis=0)
        true2 = true_masks2.transpose((0, 2, 3, 1))
        mean_iou2, iou2 = calculate_iou(true_masks2, predicted_masks)
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images, true2,pred, ix=i, number = '2')
            df2,df4 = video_sample(true2,pred,ix=i)
            diff2.append(df2)
            print('sample {} saved'.format(i))
        diff2 = np.array(diff2)
        wandb.log({"video2": wandb.Video(diff2,fps=1)})
        print("Mean IoU (annotator2):", mean_iou2)
        print("IoU dla każdej klasy (annotator2):", iou2)

    
        test_metrics =  {"inference/Mean Iou (annotator1)": mean_iou, 
                     "inference/Iou for each class (annotator1)": iou,
                     "inference/Mean Iou (annotator2)": mean_iou2,
                     "inference/Iou for each class (annotator2)": iou2,
                       }
        
    elif config.mode == 'intersection_and_union_inference':
        true_masks2 = np.concatenate(true_masks2, axis=0)
        true2 = true_masks2.transpose((0, 2, 3, 1))
        unions_numpy = np.concatenate(unions_list, axis=0)
        unions_plot = unions_numpy.transpose((0, 2, 3, 1))
        intersections_numpy = np.concatenate(intersections_list, axis=0)
        intersections_plot = intersections_numpy.transpose((0, 2, 3, 1))


        mean_iou_union, iou_union = calculate_iou(unions_numpy, predicted_masks)
        print("Mean IoU (Unia):", mean_iou_union)
        print("IoU dla każdej klasy (Unia):", iou_union)
        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,unions_plot,pred, ix=i)
            df1,df2 = video_sample(unions_plot,pred,ix=i)
            diff.append(df1)
            diff2.append(df2)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        diff2 = np.array(diff2)
        wandb.log({"video_pdiff(union)": wandb.Video(diff,fps=1)})
        wandb.log({"video_cdiff(union)": wandb.Video(diff2,fps=1)})


        mean_iou_ids1, iou_ids1 = calculate_iou(true_masks, predicted_masks)
        print("Mean IoU (annotator1):", mean_iou_ids1)
        print("IoU dla każdej klasy (annotator1):", iou_ids1)
        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,true,pred, ix=i,number = '2')
            df1,df2 = video_sample(true,pred,ix=i)
            diff.append(df1)
            diff2.append(df2)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        diff2 = np.array(diff2)
        wandb.log({"video_pdiff(ids1)": wandb.Video(diff,fps=1)})
        wandb.log({"video_cdiff(ids1)": wandb.Video(diff2,fps=1)})

        mean_iou_ids2, iou_ids2 = calculate_iou(true_masks2, predicted_masks)
        print("Mean IoU (annotator2):", mean_iou_ids2)
        print("IoU dla każdej klasy (annotator2):", iou_ids2)
        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,true2,pred, ix=i, number = '3')
            df1,df2 = video_sample(true2,pred,ix=i)
            diff.append(df1)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        wandb.log({"video_pdiff(ids2)": wandb.Video(diff,fps=1)})

        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,unions_plot-intersections_plot,pred-intersections_plot, ix=i, number = '4')
            df1 = video_sample2(intersections_plot,unions_plot,pred,ix=i)
            diff.append(df1)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        wandb.log({"video_pdiff(union-intersection)": wandb.Video(diff,fps=1)})


        test_metrics = {"inference/Mean Iou (annotator1)": mean_iou_ids1, 
                     "inference/Iou for each class (annotator1)": iou_ids1,
                     "inference/Mean Iou (annotator2)": mean_iou_ids2,
                     "inference/Iou for each class (annotator2)": iou_ids2,
                     "inference/Mean Iou (union)": mean_iou_union,
                    "inference/Iou for each class (union)": iou_union,
                       }
   
    elif config.mode == 'intersection_inference':
        true_masks2 = np.concatenate(true_masks2, axis=0)
        true2 = true_masks2.transpose((0, 2, 3, 1))
        unions_numpy = np.concatenate(unions_list, axis=0)
        unions_plot = unions_numpy.transpose((0, 2, 3, 1))
        intersections_numpy = np.concatenate(intersections_list, axis=0)
        intersections_plot = intersections_numpy.transpose((0, 2, 3, 1))


        mean_iou_intersection, iou_intersection = calculate_iou(intersections_numpy, predicted_masks)
        print("Mean IoU (Intersection):", mean_iou_intersection)
        print("IoU dla każdej klasy (Intersection):", iou_intersection)
        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,intersections_plot,pred, ix=i)
            df1,df2 = video_sample(intersections_plot,pred,ix=i)
            diff.append(df1)
            diff2.append(df2)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        diff2 = np.array(diff2)
        wandb.log({"video_pdiff(union)": wandb.Video(diff,fps=1)})
        wandb.log({"video_cdiff(intersection)": wandb.Video(diff2,fps=1)})


        mean_iou_ids1, iou_ids1 = calculate_iou(true_masks, predicted_masks)
        print("Mean IoU (annotator1):", mean_iou_ids1)
        print("IoU dla każdej klasy (annotator1):", iou_ids1)
        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,true,pred, ix=i,number = '2')
            df1,df2 = video_sample(true,pred,ix=i)
            diff.append(df1)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        wandb.log({"video_pdiff(ids1)": wandb.Video(diff,fps=1)})
        

        mean_iou_ids2, iou_ids2 = calculate_iou(true_masks2, predicted_masks)
        print("Mean IoU (annotator2):", mean_iou_ids2)
        print("IoU dla każdej klasy (annotator2):", iou_ids2)
        diff = []
        diff2 = []
        for i in range(len(x_images)):
            plot_sample(x_images,true2,pred, ix=i, number = '3')
            df1,df2 = video_sample(true2,pred,ix=i)
            diff.append(df1)
            print('sample {} saved'.format(i))
        diff = np.array(diff)
        wandb.log({"video_pdiff(ids2)": wandb.Video(diff,fps=1)})

        test_metrics = {"inference/Mean Iou (annotator1)": mean_iou_ids1, 
                     "inference/Iou for each class (annotator1)": iou_ids1,
                     "inference/Mean Iou (annotator2)": mean_iou_ids2,
                     "inference/Iou for each class (annotator2)": iou_ids2,
                     "inference/Mean Iou (intersection)": mean_iou_intersection,
                    "inference/Iou for each class (intersection)": iou_intersection,
                       }
    else:
        print("Mean IoU:", mean_iou)
        print("IoU dla każdej klasy:", iou)

        test_metrics =  {"inference/Mean Iou": mean_iou, 
                        "inference/Iou for each class": iou,
                        }
    wandb.log(test_metrics)
   
    

num_classes = 4
path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            } 

model_dict = {'myUNet': UNet(3,num_classes),
              'smpUNet': smp.Unet(in_channels = 3, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels = 3, classes=num_classes),
}

mode_dict = {'normal': 'mixed',
             'intersection': 'intersection',
             'intersection_and_union': 'intersection_and_union',
             'both': 'both',
             'intersection_and_union_inference': 'intersection_and_union_inference',
             'intersection_inference': 'intersection_inference'

}


wandb.init(project="noisy_labels", entity="segsperm",
            config={
            "model": "smpUNet++",
            "batch_size": 1,
            "annotator": 1,
            "place": 'lab',
            "mode": "intersection_and_union_inference"
            })

config = wandb.config

with open(path_dict[config.place], 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

saved_model_name = 'Annotator_1_Model_smpUNet++_Augmentation_False_Modeintersection_and_union_Optimizer_Adam_Scheduler_CosineAnnealingLR_Epochs_300_Batch_Size_22_Start_lr_0.0001_Loss_BCEWithLogitsLoss_Timestamp_2024-03-09-20-00_best_model'
model_path = yaml_config['save_model_path'] + '/' + saved_model_name
name = (f'Inference: Model_name: {saved_model_name}')

wandb.run.name = name
batch_maker = BatchMaker(config_path=path_dict[config.place], batch_size=config.batch_size, mode = 'test',segment = mode_dict[config.mode],annotator= config.annotator)
test_loader = batch_maker.test_loader


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