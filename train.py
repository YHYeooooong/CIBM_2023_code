
from cProfile import run
from matplotlib.container import BarContainer
import timm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import max_error, precision_score, recall_score, average_precision_score, roc_auc_score
from timm.loss import SoftTargetCrossEntropy

from Learningrate_scheduler import CosineAnnealingWarmUpRestarts
from writ_board import write_tensorboard


#----------------------------------------------------------------parser--------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--aug', required=True, choices=['Advanced', 'Basic'])
parser.add_argument('--LR', required=True, choices=['w_sch', 'wo_sch'])
parser.add_argument('--iter', required=True)
parser.add_argument('--num_cls', required=True, choices=['2', '4'])
parser.add_argument('--model_weight', required=True, choices=['1k', '21k'])

args = parser.parse_args()
model_name = args.model
augment_type = args.aug
LR_type = args.LR
iters = args.iter
#-------------------------------------------------setting---------------------------------------------------


torch.cuda.empty_cache()
# 디폴트 값
Epoch = 300
Weight_decay = 0.01
num_classes=args.num_cls
batch_size = 64


#------------------------------------------------------model -----------------------------------------------------------
device_num = 0
device = torch.device("cuda:"+str(device_num) if torch.cuda.is_available() else "cpu")
CCpu = torch.device('cpu')


if args.model_weight == '1k' :
    from nclass_1k import define_model
    model, clipping = define_model(model_name, num_classes, device)

else : 
    from nclass_21k import define_model
    model, clipping = define_model(model_name, num_classes, device)


#--------------------------------------------------learning rate and name ---------------------

today = time.strftime('%Y-%m-%d', time.localtime(time.time())) 

if LR_type == 'wo_sch' :
    Learning_rate = 0.0001
    optimizer_ft = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=Weight_decay)
    scheduler=0

elif LR_type == 'w_sch' :
    Learning_rate = 0.00000001 
    optimizer_ft = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=Weight_decay) 
    scheduler = CosineAnnealingWarmUpRestarts(optimizer_ft, T_0=300, T_mult=1, eta_max=0.0001,  T_up=10, gamma=0.8)

file_name = str(model_name) +'_iter'+str(iters)+ '_e'+str(Epoch)+'_opti_AdamW_'+str(LR_type)+'_LR_0.0001_bat_'+str(batch_size)+'_aug_'+str(augment_type)+'_clip'+str(clipping)+'_sampled'



#------------------------------------------save point----------------------------------------------------------------------

total_path = "../result/iter"+iters+'/'+file_name+'/' 
if not os.path.isdir("../result/iter"+iters+'/'+file_name) :
    os.mkdir("../result/iter"+iters+'/'+file_name)
writer = SummaryWriter(total_path+'/tensorboard')


#----------------------------------------------augmentation and loss-----------------------------------------------------------------------



if augment_type =='Advanced' :
    #advanced transforms
    
    #mixup, cutmix, labelsmoothing
    mixup_al = 0.2
    cutmix_al = 1.00
    cutmix_minmax = None
    prob= 0.5
    switch_prob= 0.5
    mode = 'batch'
    label_smoothing = 0.1

    mixup_args = dict(
            mixup_alpha=mixup_al, cutmix_alpha=cutmix_al, cutmix_minmax=cutmix_minmax,
            prob=prob, switch_prob=switch_prob, mode=mode,
            label_smoothing=label_smoothing, num_classes=2)
    data_transforms = {
        'train': transforms.Compose([
            
            transforms.RandomResizedCrop((224,224), interpolation=transforms.InterpolationMode.BICUBIC),     
            transforms.RandomHorizontalFlip(),    
            transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.ToTensor(), 
            transforms.RandomErasing(p=0.25)     
        ]),
        'val': transforms.Compose([
            transforms.ToTensor() 
        ]),
    }

    data_dir = '../dataset' 

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                            data_transforms['train']) 
   
    from timm.data import Mixup
    mixup_fn = Mixup(**mixup_args)
    
    criterion = SoftTargetCrossEntropy() 



elif augment_type =='Basic' :
    
    data_transforms = {
        'train': transforms.Compose([
            
            transforms.RandomResizedCrop((224,224)),    
            transforms.RandomHorizontalFlip(),    
            transforms.ToTensor()
            
        ]),
        'val': transforms.Compose([
            transforms.ToTensor() 
        ]),
    }

    data_dir = '../dataset'

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                            data_transforms['train']) 
    criterion = nn.CrossEntropyLoss() 



val_criterion = nn.CrossEntropyLoss() 
val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), 
                                        data_transforms['val']) 
image_datasets = {}
image_datasets['train'] = train_data 
image_datasets['val'] = val_data 


#------------------------------------------------------------------------------------------------------


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, 
                                            shuffle=True, num_workers=4)
                                            for x in ['train', 'val']} 
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} 





#---------------------------------------------------------------------------------------------------------------

result = {}
train_acc = []
train_loss = []
val_acc = []
val_loss = []

def train_model(model, criterion, val_criterion, optimizer, scheduler, path, writer, max_norm, num_epochs=25, result=result):
    since = time.time()
    train_time = 0
    val_time = 0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
 
    

    best_result = {}
    
    material = {}
    material['y_hat'] = []
    material['y_test'] = []
    material['y_probab'] = []

    board_result = {}
    board_result['recall'] = []
    board_result['precision'] = []
    board_result['AUC'] = []
    board_result['lr'] = []

    T_y_hat = []
    T_y_test = []
    T_y_probab = []

    T_recall = []
    T_precision = []
    T_lr = []
    T_auc = []


    #--------------------------------------------------#
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                train_start = time.time()
            elif phase == 'val' :
                model.eval()  
                val_start = time.time()

            running_loss = 0.0
            running_corrects = 0
            total_num=0

            probab = []
            y_hat = []
            y_test = []


            #-----------------------------------------------#
           
            for inputs, labels in dataloaders[phase]: 

                inputs = inputs.to(device)
                labels = labels.to(device) 
        
                
                        
                if augment_type =='Advanced' and phase == 'train' :
                    inputs, labels = mixup_fn(inputs, labels)

                
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)



                    if phase == 'train':
                        loss = criterion(outputs, labels)
                        loss.backward()

                        if max_norm != 'No' : 
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                        optimizer.step() 
                    
                    elif phase == 'val':
                        loss = val_criterion(outputs, labels)
               

             
                running_loss += loss.item() * inputs.size(0) 

                #---------------------------------------------------------------#

                
                if augment_type == 'Basic':
                    running_corrects += torch.sum(preds == labels.data) 
                    total_num += len(labels.data) 

                elif phase == 'val' : 
                    running_corrects += torch.sum(preds == labels.data) 
                    total_num += len(labels.data) 
                

                
                one_soft= nn.Softmax(dim = 1) 
                soft_output = one_soft(outputs)
                soft_outputs = soft_output.to(CCpu).tolist() 
                probab.extend(soft_outputs) 
                    

                y_hat.extend(preds.to(CCpu).tolist()) 
                y_test.extend(labels.data.to(CCpu).tolist()) 
            
            if phase == 'train' :
                if LR_type == 'w_sch':  
                    scheduler.step()
                epoch_acc = 0

            if phase == 'val' or augment_type == 'Basic' :
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                correct_num = running_corrects.to(CCpu).tolist() 
                wrong_num = total_num - correct_num 
                epoch_acc = epoch_acc.to(CCpu) 
                epoch_acc = epoch_acc.item()     

            else : 
                epoch_acc = 0

            epoch_loss = running_loss / dataset_sizes[phase] 
            



            
            epoch_loss = epoch_loss#.to(CCpu) 


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

            writer, E_y_hat, E_y_test, E_y_probab, E_recall, E_precision, E_lr, E_auc = write_tensorboard(writer, phase, epoch, epoch_acc, 
            epoch_loss, y_hat, y_test, probab, scheduler, optimizer)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_yhat = E_y_hat
                best_ytest = E_y_test
                best_y_probab = E_y_probab
                best_recall = E_recall
                best_precision = E_precision
                best_lr = E_lr
                best_AUC =  E_auc
                best_epoch = epoch
                
            if phase == 'train' :
                train_acc.append(epoch_acc) 
                train_loss.append(epoch_loss) 
                train_time += time.time() - train_start 

            elif phase == 'val' :
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

                T_y_hat.append(E_y_hat)
                T_y_test.append(E_y_test)
                T_y_probab.append(E_y_probab)

                T_recall.append(E_recall)
                T_precision.append(E_precision)
                T_lr.append(E_lr)
                T_auc.append(E_auc)
                val_time += time.time() - val_start 

            if (phase == 'val') and (epoch == num_epochs - 1) :
                val_y_hat = y_hat
                val_y_test = y_test
                val_probab = probab 
                check_cor = 0
                check_wrong = 0



        print()
    writer.close()
    result['train_acc'] = train_acc 
    result['train_loss'] = train_loss
    result['val_acc'] = val_acc
    result['val_loss'] = val_loss

    result = pd.DataFrame(result)
    result.to_csv(path + file_name +'_acc_loss.csv') 

    con_csv = {}
    con_csv['val_y_hat'] = val_y_hat
    con_csv['val_y_test'] = val_y_test
    con_csv['correct'] = correct_num
    con_csv['wrong'] = wrong_num
    con_csv['probabilities'] = probab

    con_csv = pd.DataFrame(con_csv)
    con_csv.to_csv(path + file_name +'_hat_test.csv') 

    material['y_hat'] = T_y_hat
    material['y_test'] = T_y_test
    material['y_probab'] = T_y_probab

    board_result['recall'] = T_recall
    board_result['precision'] = T_precision
    board_result['AUC'] = T_auc
    board_result['lr'] = T_lr
    
    material = pd.DataFrame(material)
    board_result =pd.DataFrame(board_result)
    material.to_csv(path+file_name+'_board_materials.csv')
    board_result.to_csv(path+file_name+'_board_result.csv') 

    time_elapsed = time.time() - since 
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))    
    

    best_result['Best_acc'] = best_acc 
    best_result['Best_loss'] = best_loss
    best_result['Best_yhat'] = best_yhat 
    best_result['Best_ytest'] = best_ytest 
    best_result['Best_y_probab'] = best_y_probab 
    best_result['Best_recall'] = best_recall 
    best_result['Best_precision'] = best_precision 
    best_result['Best_lr'] = best_lr
    best_result['Best_AUC'] = best_AUC 
    best_result['Best_epoch'] = best_epoch
    best_result['Total_time'] = str(time_elapsed // 60) +'min ' + str(time_elapsed % 60) + 'sec'
    best_result['Train_time'] = train_time
    best_result['Val_time'] = val_time
    
    best_result = pd.DataFrame(best_result)
    best_result.to_csv(path+file_name+'_best_result.csv')

    
    



    model.state_dict() 
    return best_model_wts, model, optimizer




best_model_wts, model_ft, optimizer_ft = train_model(model, criterion, val_criterion, optimizer_ft, scheduler, total_path, writer, clipping, num_epochs=Epoch, result=result) # 모델 훈련



torch.save(best_model_wts, total_path + file_name +'best_model.pt') 
torch.save(model_ft, total_path + file_name +'.pt')