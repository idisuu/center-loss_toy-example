import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

#sys.path.append("..")
from loss.center_loss import CenterLoss

# 환경에 따라 tqdm 모듈을 동적으로 로딩
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class Trainer:

    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        optimizer,
        lr: float,
        epochs: int,
        batch_size: int,
        center_loss_config = None,
        device="cuda",
        save_path = None,
        use_softmax_initially: float = 0,
        manually_update_center = False,
        
    ):
        self.model = model
        self.model.to(device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.optimizer = optimizer(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.device= device

        self.epochs = epochs
        if (save_path is not None) and (not os.path.exists(os.path.dirname(save_path))):
            raise Exception(f"folder ({os.path.dirname(save_path)}) does not exist. please check!")
        self.save_path = save_path

        self.manually_update_center = manually_update_center
        
        self.softmax_loss_fct = nn.CrossEntropyLoss()
        self.center_loss_config = center_loss_config
        if self.center_loss_config:
            use_gpu = False
            if "cuda" in self.device:
                use_gpu = True
            self.center_loss = CenterLoss(use_gpu=use_gpu)
            self.center_loss_optimizer = optimizer(self.center_loss.parameters(), lr=0.5)
            
        self.softmax_only_epoch = int(use_softmax_initially * self.epochs)
        
    def train(self):
        result_list = []
        warmup_iteration = 0
        for epoch in tqdm(range(1, self.epochs+1)):
            self.model.train()
            if epoch == int(self.epochs * 0.75):
                print(f'LR: {self.optimizer.param_groups[0]["lr"]} ==> {self.optimizer.param_groups[0]["lr"]/10}')
                for param in self.optimizer.param_groups:
                    param['lr'] /= 10
                if self.center_loss_config:
                    for param in self.center_loss_optimizer.param_groups:
                        param['lr'] /= 10
            # Manually update centers to avoid gradient explosion when adding center loss on top of existing softmax loss after a few epochs.
            if self.manually_update_center and self.softmax_only_epoch == epoch:
                new_centers = self.calc_centers()
                self.center_loss.centers.data = new_centers * 0.2
                print('centers updated manually!')
                print(new_centers)
                
            result = {}            
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                input, label = batch
                input, label = input.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                feature, out = self.model(input)    
                softmax_loss = self.softmax_loss_fct(out, label)
                if not result.get('softmax_loss'):
                    result['softmax_loss'] = 0
                result['softmax_loss'] += softmax_loss.item()
                
                
                if not self.center_loss_config or  epoch < self.softmax_only_epoch:                    
                    softmax_loss.backward()
                    self.optimizer.step()
                else:
                    warmup_iteration += 1
                    self.center_loss_optimizer.zero_grad()
                                        
                    center_loss = self.center_loss_config['alpha'] * self.center_loss(feature, label)

                    alpha = 0.01
                    warmup_rate = min(1, np.exp(alpha * warmup_iteration) / np.exp(alpha*self.center_loss_config['warmup_steps']))
                    #print(warmup_rate)
                    total_loss_without_warmup = softmax_loss + self.center_loss_config["lambda"] * center_loss
                    total_loss = softmax_loss + warmup_rate * self.center_loss_config["lambda"] * center_loss
                    
                    #print(softmax_loss.item(), center_loss.item(), total_loss.item())

                    if torch.isnan(total_loss):
                        raise Exception('The loss became nan, so finished the training')
                        

                    if not result.get('center_loss'):
                        result['center_loss'] = 0
                    if not result.get('total_loss'):
                        result['total_loss'] = 0

                    result['center_loss'] += center_loss.item()
                    result['total_loss'] += total_loss_without_warmup.item()
                    
                    total_loss.backward()
                    
                    for param in self.center_loss.parameters():
                        param.grad.data *= (1. / self.center_loss_config["lambda"])
                    self.center_loss_optimizer.step()
                    self.optimizer.step()

            for key in result.keys():
                result[key] /= len(self.train_loader)            

            train_acc = self.validate(mode='train')
            test_acc = self.validate(mode='test')
            print('='*50)
            print(f'Epoch: {epoch}')
            print(f"train_acc: {train_acc['accuracy']}")
            print(f"test_acc: {test_acc['accuracy']}")
            print(result)
            print('='*50)

            result['train_acc'] = train_acc['accuracy']
            result['test_acc'] = test_acc['accuracy']
            
            result_list.append(result)
            
        if self.save_path:
            torch.save(self.model, self.save_path)
            print(f"Model saved in {self.save_path}")

        return result_list
        
    def validate(
        self,
        mode='test',
        checkpoint=None,
        visualize=None,
    ):
        if not checkpoint:
            validation_model = self.model
        else:
            validation_model = torch.load(checkpoint)

        validation_model.eval()

        if mode == 'train':
            dataloader = self.train_loader
        elif mode == 'test':
            dataloader = self.test_loader
        else:
            raise Exception(f"mode should be 'train' or 'test', but your input was {mode}")

        total = 0
        correct = 0
        if visualize:
            feature_dict = {i: ([], []) for i in range(10)}
            
        for batch in tqdm(dataloader, total=len(dataloader)):
            input, label = batch
            input, label = input.to(self.device), label
            
            with torch.no_grad():
                feature, out = validation_model(input)
                
            pred = torch.argmax(out, dim=-1).cpu().numpy()
            label = label.numpy()    

            batch_correct = (pred == label).sum()
            correct += batch_correct
            total += len(label)
            
            if visualize:
                for p, f in zip(pred, feature):
                    f = list(f.cpu().numpy())
                    feature_dict[p][0].append(f[0])
                    feature_dict[p][1].append(f[1])            

        acc = correct / total
        
        if visualize:
            plt.figure(figsize=(20, 20))
            for i in range(10):
                plt.scatter(feature_dict[i][0], feature_dict[i][1], label=i)
                plt.legend()
            plt.show()
                
        validation_model.train()
        
        return {'accuracy': acc}

    def calc_centers(self, mode='train'):
        self.model.eval()
        if mode == 'train':
            dataloader = self.train_loader
        elif mode == 'test':
            dataloader = self.test_loader
        else:
            raise Exception(f"mode should be 'train' or 'test', but your input was {mode}")

        centers = torch.zeros(self.center_loss.num_classes, self.center_loss.feat_dim).to(self.device)

        for batch in tqdm(dataloader, total=len(dataloader)):
            input, label = batch
            input, label = input.to(self.device), label
            with torch.no_grad():
                feature, out = self.model(input)

            for i in range(self.center_loss.num_classes):
                mask = (label == i)
                mean = feature[mask].mean(dim=0)
                centers[i] += mean
                
        centers /= len(dataloader)

        self.model.train()
        return centers
                
            













