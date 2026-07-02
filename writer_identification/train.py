"""
This code is edited from the following work:

    This code is for the following paper:
    
    Sheng He and Lambert Schomaker
    GR-RNN: Global-Context Residual Recurrent Neural Networks for Writer Identification
    Pattern Recognition
    
    @email: heshengxgd@gmail.com
    @author: Sheng He
    @Github: https://github.com/shengfly/writer-identification
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random
import dataloader_binarised as bindset
import dataloader_real as dset

import GRRNN as net
import numpy as np
import os
import argparse

class LabelSomCE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x,target,smoothing=0.1):
		confidence = 1.0 - smoothing
		logprobs = F.log_softmax(x,dim=-1)
		nll_loss = - logprobs.gather(dim=-1,index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = confidence * nll_loss + smoothing * smooth_loss

		return loss.mean()
def set_seed(seed=42):
    random.seed(seed)                        # Python random
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # CPU ops
    torch.cuda.manual_seed(seed)             # GPU ops
    torch.cuda.manual_seed_all(seed)         # multi-GPU
    torch.backends.cudnn.deterministic = True  # deterministic conv ops
    torch.backends.cudnn.benchmark = False   # disable auto-tuner

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DeepWriter_Train:
    def __init__(self, dataset='CVL', imgtype='png', mode='vertical', seed=42,subfolder=None,tfolder=None,
                 train_folder=None,image_type='binarised',dataset_path='', print_label='', batch_size=16):
        set_seed(seed)
        self.dataset = dataset
        self.image_type=image_type
        self.dataset_path = dataset_path
        self.labelfolder = dataset_path + subfolder
        self.folder = dataset_path + subfolder
        if not os.path.exists(self.folder):
            print('****** Warning: the dataset %s does not existed!******'%dataset)
            print('Please go to the following website to check how to download the dataset:')
            print('https://www.ai.rug.nl/~sheng/writeridataset.html')
            print('*'*20)
            raise ValueError('Dataset: %s does not existed!'%dataset)
        
        self.labelfolder = self.folder
        self.train_folder = self.folder + train_folder
        self.test_folder = self.folder + tfolder
        print(print_label if print_label else "real data and 200 percent of generated data ")
        self.imgtype=imgtype
        self.mode = mode
        self.device = 'cuda'
        self.scale_size=(64,128)
        # if self.device == 'cuda':
        #     torch.backends.cudnn.benchmark = True
        if self.dataset == 'CVL':
            self.imgtype = 'tif'
        elif self.dataset == 'bullinger':
            self.imgtype = 'png'
            
        if self.image_type == 'binarised':
            dt = bindset
        elif self.image_type == 'grayscale':
            dt=dset
            
        self.model_dir = 'model'
        
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        basedir = 'GRRNN_WriterIdentification_dataset_'+self.dataset+'_model_'+self.mode+'_aug_16'
        self.logfile= basedir + '.log'
        self.modelfile = basedir
        self.batch_size = batch_size
        train_set = dt.DatasetFromFolder(dataset=self.dataset,
        				labelfolder = self.labelfolder,
                        foldername=self.train_folder,
                        imgtype=self.imgtype,
                        scale_size=self.scale_size,
                        is_training = True)
        
        test_set = dt.DatasetFromFolder(dataset=self.dataset,
        				labelfolder = self.labelfolder,
                        foldername=self.test_folder,imgtype=self.imgtype,
                        scale_size=self.scale_size,
                        is_training = False)
        
        g = torch.Generator()
        g.manual_seed(seed)

        self.training_data_loader = DataLoader(
            dataset=train_set,
            num_workers=6,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            persistent_workers=True,
        )

        self.testing_data_loader = DataLoader(
            dataset=test_set,
            num_workers=6,
            batch_size=self.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            pin_memory=True,
            persistent_workers=True,
        )
        
        num_class = train_set.num_writer
        self.model = net.GrnnNet(1,num_classes=train_set.num_writer,mode=self.mode).to(self.device)
        
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = LabelSomCE()
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001,weight_decay=1e-4) 
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.5)
                
    def train(self,epoch):
        self.model.train()
        losstotal = []
        
        for iteration,batch in enumerate(self.training_data_loader,1):
            inputs = batch[0].to(self.device).float()
            target = batch[1].type(torch.long).to(self.device)
        
            self.optimizer.zero_grad()
   
            logits = self.model(inputs)
         
            train_loss= self.criterion(logits,target)

            losstotal.append(train_loss.item())
            train_loss.backward()
            self.optimizer.step()
        
        with open(self.logfile,'a') as fp:
            fp.write('Training epoch %d avg loss is: %.6f\n'%(epoch,np.mean(losstotal)))
        print('Traing epoch:',epoch,'  avg loss is:',np.mean(losstotal))

    def test(self,epoch,during_train=True):
        self.model.eval()
        
        if not during_train:
            self.load_model(epoch)

        top1 = 0
        top5 = 0
        ntotal = 0

        # -------- Per-writer counters --------
        writer_top1 = {}
        writer_top5 = {}
        writer_total = {}
        
        for iteration,batch in enumerate(self.testing_data_loader,1):
            inputs = batch[0].to(self.device).float()
            target = batch[1].to(self.device).long()
        
            logits = self.model(inputs)
            
            # -------- Global accuracy (unchanged) --------
            res = self.accuracy(logits,target,topk=(1,5))
            top1 += res[0]
            top5 += res[1]
            
            ntotal += inputs.size(0)

            # -------- Per-writer accuracy --------
            _, pred = logits.topk(5,1,True,True)

            for i in range(inputs.size(0)):
                writer_id = target[i].item()

                if writer_id not in writer_top1:
                    writer_top1[writer_id] = 0
                    writer_top5[writer_id] = 0
                    writer_total[writer_id] = 0

                writer_total[writer_id] += 1

                # top1 check
                if pred[i,0].item() == writer_id:
                    writer_top1[writer_id] += 1

                # top5 check
                if writer_id in pred[i].cpu().numpy():
                    writer_top5[writer_id] += 1

        # -------- Global results --------
        top1 /= float(ntotal)
        top5 /= float(ntotal)

        print('Testing on epoch: %d has accuracy: top1: %.2f top5: %.2f'
            %(epoch,top1*100,top5*100))

        with open(self.logfile,'a') as fp:
            fp.write('Testing epoch %d accuracy is: top1: %.2f top5: %.2f\n'
                    %(epoch,top1*100,top5*100))

    def check_exists(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        return os.path.exists(model_out_path)
    
    def checkpoint(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        torch.save(self.model.state_dict(),model_out_path)
    
    def load_model(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model_epoch_{}.pth'.format(epoch)
        self.model.load_state_dict(torch.load(model_out_path,map_location=self.device))
        print('Load model successful')
                
    def train_loops(self,start_epoch,num_epoch):
        #if self.check_exists(num_epoch): return
        if start_epoch > 0:
            self.load_model(start_epoch-1)
        
        for epoch in range(start_epoch,num_epoch):
            self.train(epoch)
            self.checkpoint(epoch)
            if epoch % 10 == 0 or epoch == num_epoch - 1:
                self.test(epoch)
            self.scheduler.step()
        
    def accuracy(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            _,pred = output.topk(maxk,1,True,True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum()
                res.append(correct_k.data.cpu().numpy())
        
        return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--dataset', type=str, default='CVL', help='Dataset CVL or bullinger')
    parser.add_argument('--image_type', type=str, default='binarised', help='Dataloader images type grayscale or binarised')
    parser.add_argument('--subfolder', type=str, default='', help='Path to a subfolder if exists')
    parser.add_argument('--train_folder', type=str, default='/train/', help='trainset folder name')
    parser.add_argument('--tfolder', type=str,default='/test/', help='testset folder name')
    parser.add_argument('--print_label',  type=str, default='',    help='Label printed at startup')
    parser.add_argument('--batch_size',   type=int, default=16,    help='Batch size')
    parser.add_argument('--seed',   type=int, default=42,    help='seed ')
    args = parser.parse_args()

    modelist = ['vertical','horzontal']
    mode = modelist[0]
    mod = DeepWriter_Train(dataset=args.dataset, mode=mode, seed=args.seed,subfolder=args.subfolder,
                           train_folder=args.train_folder,
                           tfolder=args.tfolder,dataloader=args.dataloader,dataset_path=args.dataset_path,
                           print_label=args.print_label,
                           batch_size=args.batch_size)
    mod.train_loops(0,150)
