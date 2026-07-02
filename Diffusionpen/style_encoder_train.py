import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
from os.path import isfile
from skimage import io
from torchvision.utils import save_image
from skimage.transform import resize
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
# from utils.iam_dataset import IAMDataset
from utils.cvl_dataset import CVLDataset
from utils.bullinger_dataset import BullingerDataset
from utils.auxilary_functions import affine_transformation
from feature_extractor import ImageEncoder
import timm
import cv2
import time
import json
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def stratified_split_by_writer(dataset, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    writer_to_indices = defaultdict(list)
    for i, (img_path, transcr, wid, path) in enumerate(dataset.data):
        writer_to_indices[wid].append(i)
    train_indices = []
    val_indices   = []
    for wid in sorted(writer_to_indices.keys()):
        indices  = writer_to_indices[wid]
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_indices   += shuffled[:n_val]
        train_indices += shuffled[n_val:]
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset   = torch.utils.data.Subset(dataset, val_indices)
    return train_subset, val_subset

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def image_resize_PIL(img, height=None, width=None):
    if height is None and width is None:
        return img  # No resizing needed

    original_width, original_height = img.size

    if height is not None and width is None:
        scale = height / original_height
        new_width = int(original_width * scale)
        new_height = height
    elif width is not None and height is None:
        scale = width / original_width
        new_width = width
        new_height = int(original_height * scale)
    else:
        new_width = width
        new_height = height

    # Resize the image
    resized_img = img.resize((new_width, new_height))
    #resized_img.save('res.png')
    return resized_img


def centered_PIL(word_img, tsize, centering=(.5, .5), border_value=None):
    
    height = tsize[0]
    width = tsize[1]
    #print('word_img.size', word_img.size)
    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.height
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.height - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.width
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.width - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    
    
   
    #print('word_img.size, padw, padh', word_img.size, padw, padh)
    res = Image.new('RGB', (width, height), color = (255, 255, 255))
    #res.save('background.png')
    
    res.paste(word_img, (padw[0], padh[0]))
    
    
    return res

    
    
class Mixed_Encoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=339, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool=""
        )
        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Create the classifier
        if hasattr(self.model, 'num_features'):
            num_features = self.model.num_features
        else:
            # Fallback, can be adjusted based on the specific model
            num_features = 2048
        # self.dropout = nn.Dropout(p=0.3)      #Buliing
        self.classifier = nn.Linear(num_features, num_classes)
        for p in self.model.parameters():
            p.requires_grad = trainable
            
    def forward(self, x):
        # Extract features
        features = self.model(x)
        # Pool the features to make them of fixed size
        pooled_features = self.global_pool(features).flatten(1)
        # pooled_features = self.dropout(pooled_features) #bulling
        # Classify
        logits = self.classifier(pooled_features)
        return logits, pooled_features  


#================ Performance and Loss Function ========================
def performance(pred, label):  
    loss = nn.CrossEntropyLoss()
    loss = loss(pred, label)
    return loss 

#===================== Training ==========================================

def train_class_epoch(model, training_data, optimizer, args):
    '''Epoch operation in training phase'''
    
    model.train()
    total_loss = 0
    n_corrects = 0 
    total = 0
    pbar = tqdm(training_data)
    for i, data in enumerate(pbar):
    
        image = data[0].to(args.device)
        if args.dataset == 'iam':
            label = data[2].to(args.device)
        
        optimizer.zero_grad()

        output = model(image)
        
        loss = performance(output, label)
        _, preds = torch.max(output.data, 1)
 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        total += label.size(0)
        n_corrects += (preds == label).sum().item()
        pbar.set_postfix(Loss=loss.item())
        
    loss = total_loss/total
    accuracy = n_corrects/total
    
    return loss, accuracy

def eval_class_epoch(model, validation_data, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0
    prediction_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_data)):

            image = data[0].to(args.device)   
            image_paths = data[4]
            if args.dataset == 'iam':
                label = data[2].to(args.device)

            output = model(image)
            
            loss = performance(output, label)  #performance
            _, preds = torch.max(output.data, 1)
            
            total_loss += loss.item()
            n_corrects += (preds == label.data).sum().item()
            total += label.size(0)
            #prediction_list.append(preds)
            #write into a file the img_path and the prediction
            # with open('predictions.txt', 'a') as f:
            #     for i, p in enumerate(preds):
            #         f.write(f'{image_paths[i]},{p}\n')
            
    loss = total_loss/total
    accuracy = n_corrects/total

    return loss, accuracy




########################################################################              
def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
    
        wid = data[2]
        #print('wid', wid)
        positive = data[3]
        negative = data[4]
        
        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(triplet_loss=loss_meter.avg)
        total += img.size(0)
    
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    return running_loss/total #np.mean(running_loss)/total

def val_epoch_triplet(val_loader, model, criterion, optimizer, device, args):
    
    running_loss = 0
    total = 0
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        #transcr = data[1]

        if args.dataset == 'iam':
            wid = data[2]
            positive = data[3]
            negative = data[4]
       
        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
    
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        pbar.set_postfix(triplet_loss=loss.item())
        total += wid.size(0)
    
    print('total', total)
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    return running_loss/total #np.mean(running_loss)/total



############################ MIXED TRAINING ############################################              
def train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        wid = data[3].to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img.to(device)
        # Get logits and features from the model
        #out = model(anchor)
        #print(type(out), len(out))
        
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        
        loss = classification_loss + triplet_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg, classification_loss=loss_meter_class.avg, triplet_loss=loss_meter_triplet.avg)
        total += img.size(0)
    
    accuracy = n_corrects/total
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    print("Training Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total

def val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0].to(device)
        wid = data[3].to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        loss = classification_loss + triplet_loss
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg)
        total += wid.size(0)
    
    print('total', total)
    accuracy = n_corrects/total
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    print("Validation Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total

#TRAINING CALLS

def train_mixed(model, train_loader, val_loader, criterion_triplet, criterion_classification, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/mixed_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
        
        
def train_classification(model, training_data, validation_data, optimizer, scheduler, device, args): #scheduler # after optimizer
    ''' Start training '''

    valid_accus = []
    num_of_no_improvement = 0
    best_acc = 0
    
    for epoch_i in range(args.epochs):
        print('[Epoch', epoch_i, ']')

        start = time.time()
        #wandb.log({'lr': scheduler.get_last_lr()})
        #print('Epoch:', epoch_i,'LR:', scheduler.get_last_lr())

        train_loss, train_acc = train_class_epoch(model, training_data, optimizer, args)
        print('Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100*train_acc,
                  elapse=(time.time()-start)/60))
        
        start = time.time()
        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}

        if validation_data is not None:
            val_loss, val_acc = eval_class_epoch(model, validation_data, args)
            print('Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                        loss=val_loss, accu=100*val_acc,
                    elapse=(time.time()-start)/60))
            
            if val_acc > best_acc:
                
                print('- [Info] The checkpoint file has been updated.')
                best_acc = val_acc
                torch.save(model.state_dict(), f"{args.save_path}/{args.dataset}_classification_{args.model}.pth")
                num_of_no_improvement = 0
            else:
                num_of_no_improvement +=1
            
        
            if num_of_no_improvement >= 10:
                        
                print("Early stopping criteria met, stopping...")
                break
        else:
            torch.save(model.state_dict(), f"{args.save_path}/{args.dataset}_classification_{args.model}.pth")

        scheduler.step()
        #wandb.log({'epoch': epoch_i, 'train loss': train_loss, 'val loss': val_loss})
        #wandb.log({'epoch': epoch_i, 'train acc': 100*train_acc, 'val acc': 100*val_acc})
        

def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_triplet(val_loader, model, criterion, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/triplet_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
        
        

def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Train Style Encoder')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--dataset', type=str, default='cvl', help='dataset name  /cvl or bullinger or you can edit your own')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training')
    parser.add_argument('--dataset_fold', type=str, default='') 
    parser.add_argument('--epochs', type=int, default=20, required=False, help='number of training epochs')
    parser.add_argument('--pretrained', type=bool, default=False, help='use of feature extractor or not')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--save_path', type=str, default='./style_models', help='path to save models')
    parser.add_argument('--mode', type=str, default='mixed', help='mixed for DiffusionPen, triplet for DiffusionPen-triplet, or classification for DiffusionPen-triplet')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    #========= Data augmentation and normalization for training =====#
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)
    
    if args.dataset == 'cvl':
        print('loading CVL')
        myDataset = CVLDataset
    elif args.dataset == 'bullinger':
        print('loading Bullinger')
        myDataset = BullingerDataset 
        
    dataset_folder=args.dataset_fold
    aug_transforms = [lambda x: affine_transformation(x, s=.1)]
        
    train_transform = transforms.Compose([
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])
        
    val_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])
        
    train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), transforms=train_transform)
        
    #print('len train data', len(train_data))
        
    # Use random_split to split the dataset into train and validation sets
    train_data, val_data = stratified_split_by_writer(train_data_full, val_ratio=0.2, seed=args.seed)
    print('len train data', len(train_data))
    print('len val data',   len(val_data))
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=32,worker_init_fn=seed_worker)
    val_loader   = DataLoader(val_data, batch_size=args.batch_size,shuffle=False, num_workers=32,worker_init_fn=seed_worker)
    
    if val_loader is not None:
        print('Val data')
    else:
        print('No validation data')
    style_classes = train_loader.dataset.dataset.num_classes
    print('writer classes are :',style_classes)
 
    
    if args.model == 'mobilenetv2_100':
        print('Using mobilenetv2_100')
        model = Mixed_Encoder(model_name='mobilenetv2_100', num_classes=style_classes, pretrained=True, trainable=True)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        # if args.pretrained == True:
        #     PATH = '/HOME/yazbeckh/DiffusionPen/style_models/iam_style_diffusionpen.pth'
        #     state_dict = torch.load(PATH, map_location=args.device)
        #     model_dict = model.state_dict()
        #     state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        #     model_dict.update(state_dict)
        #     model.load_state_dict(model_dict)
        #     #print(model)
        #     print('Pretrained mobilenetv2_100 model loaded')
            
    if args.model == 'resnet18':
        print('Using resnet18')
        model = Mixed_Encoder(model_name=args.model, num_classes=style_classes, pretrained=True, trainable=True)
        print('Model loaded')
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        # if args.pretrained == True:
        #     PATH = ''
        #     state_dict = torch.load(PATH, map_location=args.device)
        #     model_dict = model.state_dict()
        #     state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        #     model_dict.update(state_dict)
        #     model.load_state_dict(model_dict)

    model = model.to(device)
    #print(model)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    # if args.pretrained == True:
    #     optimizer_ft = optim.Adam([
    #         {'params': model.model.parameters(), 'lr': 1e-5},  # backbone — very small
    #         {'params': model.global_pool.parameters(), 'lr': 1e-4},
    #         {'params': model.classifier.parameters(), 'lr': 1e-4}  # head — larger
    #     ], weight_decay=1e-3)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer_ft, T_max=args.epochs, eta_min=1e-6
    #     )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", patience=3, factor=0.1
    )
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    #Bulling
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer_ft, mode='min', patience=5, factor=0.5, min_lr=1e-6
    # )
    # # Triplet margin smaller
    # criterion_triplet = nn.TripletMarginLoss(margin=0.5, p=2)

    #THIS IS THE CONDITION FOR DIFFUSIONPEN
    if args.mode == 'mixed':
        criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2) 
        print('Using both classification and metric learning training')
        train_mixed(model, train_loader, val_loader, criterion_triplet, None, optimizer_ft, scheduler, device, args)
        print('finished training')
    
    
    if args.mode == 'triplet':
        train(model, train_loader, val_loader, criterion, optimizer_ft, lr_scheduler, device, args)
        print('finished training')
    
    
    elif args.mode == 'classification':
        
        train_classification(model, train_loader, val_loader, optimizer_ft, scheduler, device, args)
        print('finished training')
    
    
if __name__ == '__main__':
    main()
