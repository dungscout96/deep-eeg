import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import datetime

class VGGFinetune(nn.Module):
    def __init__(self, 
                 nclass=2,               # number of classification classes
                 weights='DEFAULT',      # path to saved model weights       
                 n_freezelayers=None,    # number of layers to freeze weights. If float, it's percentage of num layer
                ):
        super().__init__()
        self.model = torchmodels.vgg16(weights=weights)
        self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-1]) 
        self.model = nn.Sequential(
            self.model,
            nn.Linear(4096, nclass),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        
        if n_freezelayers:
            if type(n_freezelayers) == float:
                n_freezelayers = len(list(self.model.named_parameters()))*n_freezelayers
            for l, (name, param) in enumerate(self.model.named_parameters()):
                if l > n_freezelayers:
                    break
                param.requires_grad = False
    
    def _input_augment_fn(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(dtype=torch.float, device=self.device)
        x = torch.reshape(x, (x.shape[0]*x.shape[1],*x.shape[2:])) # collapse time dimension
        if x.shape[-1] == 3:
            # channel should not be last dimension
            x = torch.permute(x, (0,3,1,2))

        if x.shape[-1] != 224 and x.shape[-2] != 224: 
            x = torch.nn.functional.interpolate(x, size=(224,224))
        return x

    def forward(self, x):
        x = self._input_augment_fn(x)
        x = self.model(x)   
        return x
            
    def loss(self, z, label):
        label = label.squeeze().to(device=self.device)
        z = z.to(device=self.device)
        loss = F.cross_entropy(z, label.to(dtype=torch.long))
        return loss
    
    def train(self, 
              loader_train,
              lr=0.01,
              optimizer=None,
              epochs=100,
              save_every=None, 
              checkpoint_path='./checkpoints',
              writer=None):
        
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
            
        if not optimizer:
            optimizer = optim.Adam(self.model.parameters(), lr = lr)
            
        self.model.train()
        for e in range(epochs):
            epoch_loss = 0
            for t, (sample, label) in enumerate(loader_train):  
                if len(label.shape) == 2 and label.shape[1] == 1 and label.shape[1] != sample.shape[1]:
                    label = np.repeat(label, sample.shape[1], axis=1)  # expand T dim of label
                    label = np.reshape(label,(-1,1)) # collapse T into batch size

                z = self(sample)                 
                optimizer.zero_grad()
                loss = self.loss(z, label)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                if writer:
                    writer.add_scalar("Loss/train", loss.item(), t) # batch step

                del label
                del z
                del loss
            
            print("epoch {}:".format(e)) # iteration
            print('loss={}'.format(epoch_loss/len(loader_train)))
            print()
            
            if save_every:
                # Save model save_every epoch
                if e > 0 and e % save_every == 0:
                    torch.save(self.model.state_dict(), f"{checkpoint_path}/epoch_{e}")
        # save final model
        now = datetime.datetime.now()
        timestamp = now.strftime("%y%m%d%H%M%S")
        torch.save(self.model.state_dict(), f"{checkpoint_path}/{timestamp}")
