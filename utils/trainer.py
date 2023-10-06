import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

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
        use_center_loss = None,
        device="cuda",
        save_path = None
    ):
        self.model = model
        self.model.to(device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.device= device

        self.epochs = epochs
        if (save_path is not None) and (not os.path.exists(os.path.dirname(save_path))):
            raise Exception(f"folder ({os.path.dirname(save_path)}) does not exist. please check!")
        self.save_path = save_path
        
        self.softmax_loss_fct = nn.CrossEntropyLoss()
        self.use_center_loss = use_center_loss
        if self.use_center_loss:
            pass

    def train(self):        
        for epoch in tqdm(range(1, self.epochs+1)):
            self.model.train()
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                input, label = batch
                input, label = input.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                feature, out = self.model(input)    
                softmax_loss = self.softmax_loss_fct(out, label)        
                softmax_loss.backward()
                self.optimizer.step()
            train_acc = self.validate(mode='train')
            test_acc = self.validate(mode='test')
            print('='*50)
            print(f'Epoch: {epoch}')
            print(f"train_acc: {train_acc['accuracy']}")
            print(f"test_acc: {test_acc['accuracy']}")
            print('='*50)            
            
        if self.save_path:
            torch.save(self.model, self.save_path)
            print(f"Model saved in {self.save_path}")

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
            raise Exception(f"mode should be 'train' or 'test', but your input was {loader}")

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
            validation_model.train()
        
        return {'accuracy': acc}
                
            













