import torch
import torchvision.models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import time

def vgg16_augment(model, nclass):
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1], nn.Linear(4096, nclass))
    return model
def inception_augment(model, nclass):
    return model
    # model = torch.nn.Sequential(*list(model.children())[:-3], nn.Linear(2048, nclass))
    # return model

def sum_weights(model):
    weights_magnitude = 0
    for l, (name, param) in enumerate(model.named_parameters()):
        weights_magnitude += torch.sum(param) # torch sum all elements
    return weights_magnitude

class AE(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        vgg16 = torchmodels.vgg16(weights='DEFAULT')
        vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])
        # self.encoder = vgg16
        decoder = self._build_decoder_vgg16(vgg16)
        model = nn.Sequential(vgg16, *decoder)
        
        self.encoder = model[0]
        self.decoder = model[1:]
        sum_params = 0
        for param in self.parameters():
            sum_params += torch.sum(param)
        print(sum_params)
        if params:
            if 'checkpoint' in params:
                print('Loading weights from checkpoint')
                self.load_state_dict(torch.load(params['checkpoint']))
        sum_params = 0
        for param in self.parameters():
            sum_params += torch.sum(param)
        print(sum_params)
            
    def _build_decoder_vgg16(self, encoder):
        decoder = []
        prev_chan_dim = 512
        for idx in range(len(list(encoder.children()))-1, -1, -1):
            module = list(encoder.children())[idx]
            if type(module) == nn.Sequential:
                for c in range(len(module)-1, -1, -1):
                    child = module[c]
                    if type(child) == nn.Conv2d:
                        prev_chan_dim = child.in_channels
                    decoder.extend(self._invert_layer_vgg16(child, prev_chan_dim))
            else:
                if type(module) == nn.Conv2d:
                    prev_chan_dim = module.in_channels
                decoder.extend(self._invert_layer_vgg16(module, prev_chan_dim))
        return decoder
    
    def _invert_layer_vgg16(self, layer, in_chan):
        chan_dim = None
        if type(layer) == nn.Conv2d:
            chan_dim = layer.in_channels
            return [nn.ConvTranspose2d(layer.out_channels, layer.in_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding), nn.ReLU()]
        elif type(layer) == nn.Linear:
            return [nn.Linear(layer.out_features, layer.in_features), nn.ReLU(), nn.Dropout()]
        elif type(layer) == nn.MaxPool2d:
            return [nn.ConvTranspose2d(in_chan, in_chan, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)]
        elif type(layer) == nn.AdaptiveAvgPool2d:
            return [nn.Unflatten(1, (512, 7, 7))]
        else:
            return []
    
    # def transform(self, x):
    #     return self.encoder(x)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class AEClassifier(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        ae = AE(params)
        self.encoder = ae.encoder
        self.classifier = nn.Linear(4096,2)  
        # for param in self.classifier.parameters():
        #     torch.nn.init.ones_(param)
        #     print(param)
    def forward(self, x):
        # return self.model(x)
        x = self.encoder(x)
        # print(x)
        x = self.classifier(x)
        # print(x)
        return x

class AETransformer(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        ae = AE(params)
        self.encoder = ae.encoder
        # self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-2])
               
    def forward(self, x):
        return self.encoder(x)
    
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 8064
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 32), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*504, 2)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.reshape(-1, 4*2*504)
        x = F.sigmoid(self.fc1(x))
        return x

class BaseDataset(Dataset):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return len(self.Y)
    
class Finetuning():
    def __init__(self, 
                 model='vgg16',          # ODE model to finetune
                 nclass=2,               # number of classification classes
                 model_params = {
                     'weights':'DEFAULT',      # path to saved model weights   
                 },
                 n_freezelayers=None,    # number of layers to freeze weights. If float, it's percentage of num layer
                 seed=0,
                ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model_name = model
        if model == 'EEGNET' or model == 'AE' or model == 'AEClassifier' or model == 'AETransformer':
            self.model = globals()[model](model_params)
        else:
            self.model = getattr(torchmodels, model)(**model_params)
            if model == 'inception_v3':
                self.model = inception_augment(self.model, nclass)
            else:
                self.model = vgg16_augment(self.model, nclass)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        
        if n_freezelayers:
            if type(n_freezelayers) == float:
                n_freezelayers = len(list(self.model.named_parameters()))*n_freezelayers
            elif type(n_freezelayers) == int:
                if n_freezelayers < 0:
                    n_freezelayers = len(list(self.model.named_parameters())) + n_freezelayers - 1 
            for l, (name, param) in enumerate(self.model.named_parameters()):
                if l > n_freezelayers:
                    break
                param.requires_grad = False
        
    def get_dataloader(self, data, bs, shuffle=True):
        if type(data) == list:
            X = [i[0] for i in data]
            Y = [i[1] for i in data]
            loader = DataLoader(BaseDataset(X, Y), batch_size = bs, shuffle = shuffle)
        elif isinstance(data, Dataset):
            loader = DataLoader(data, batch_size = bs, shuffle = shuffle)
        elif type(data) == DataLoader:
            loader = data
        else:
            raise ValueError('Not accepted dataset type')
        return loader
    
    def _input_augment_fn(self, x, label):                    
        # print('x shape', x.shape)
        if self.model_name == 'EEGNET':
            x = x.to(dtype=torch.float, device=self.device).unsqueeze(1)
        else:
            if len(x.shape) == 6 and x.shape[1] == 1:
                x = np.squeeze(x, 1)
            if len(x.shape) == 5 and label.shape[0] == x.shape[0]:
                # print('Warning: sample has T dim but label does not, expanding T dim for label')
                label = np.repeat(label, x.shape[1], axis=1)  # expand T dim of label
                label = np.reshape(label,(-1,1)) # collapse T into batch size
                        
            if not torch.is_tensor(x):
                x = torch.tensor(x)
            x = x.to(dtype=torch.float, device=self.device)
            
            if len(x.shape) == 5:
                x = torch.reshape(x, (x.shape[0]*x.shape[1],*x.shape[2:])) # collapse time dimension
            if x.shape[-1] == 3:
                # channel should not be last dimension
                x = torch.permute(x, (0,3,1,2))
            if x.shape[-1] != 224 and x.shape[-2] != 224: 
                x = torch.nn.functional.interpolate(x, size=(224,224))
            # print('final x and label shape', x.shape, label.shape)
        return x, label

    def forward(self, x):
        # x = self._input_augment_fn(x)
        # print('forward x shape',x.shape)
        x = self.model(x)   
        return x
    
    def _transform_x(self, x):
        # x = self._input_augment_fn(x)
        
        self.model.eval()
        z = self.model[0].features(x)
        z = self.model[0].avgpool(z)
        z = torch.nn.Flatten()(z)
        for i, (name,module) in enumerate(self.model[0].classifier.named_children()):
            if i > 3:
                break
            z = module(z)
        return z
    
    def transform(self, dataset, bs=32):
        if type(dataset) == list:
            X = [i[0] for i in dataset]
            Y = [i[1] for i in dataset]
            loader = DataLoader(BaseDataset(X, Y), batch_size = bs, shuffle = True)
        elif isinstance(dataset, Dataset):
            loader = DataLoader(dataset, batch_size = bs, shuffle = True)
        elif type(dataset) == DataLoader:
            loader = dataset
        else:
            raise ValueError('Not accepted dataset type')
        
        X_transformed = []
        for t, (sample, label) in enumerate(loader):
            X = self._transform_x(sample)
            X_transformed.append(X.numpy(force=True))
            
            del X
        X_transformed = np.concatenate(X_transformed, axis=0)
        return X_transformed
    
    def score(self, data):
        dataloader = self.get_dataloader(data, bs=16)
        
        self.model.eval()
        results = []
        for t, (sample, label) in enumerate(dataloader): 
            sample, label = self._input_augment_fn(sample, label)
            with torch.no_grad():
                z = self.model(sample) 
            results.extend((torch.argmax(z, dim=1).numpy(force=True) == label.numpy(force=True).T)[0])

        return np.sum(np.array(results))/len(results)
    
    def loss(self, z, label):
        if self.model_name == 'AE':
            label = label.to(device=self.device)
            z = z.to(device=self.device)
            loss = F.mse_loss(z, label)
        else:
            label = label.squeeze().to(device=self.device)
            z = z.to(device=self.device)
            loss = F.cross_entropy(z, label.to(dtype=torch.long))
        del label
        del z
        return loss
    
    def train(self, 
              train_data,
              bs=16,                # batch size 
              lr=0.01,              # learning rate
              optimizer=None,
              start_from=0,
              epochs=1000,
              save_every=None, 
              checkpoint_path='./checkpoints',
              log_dir='./runs',
              early_stopping_eps=0.1,
              lr_decay_nepoch=100):
        
        loader_train = self.get_dataloader(train_data, bs)

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
            
        if not optimizer:
            optimizer = optim.SGD(self.model.parameters(), lr = lr)
        
        if not os.path.exists(log_dir):
            try:
                os.mkdir(log_dir)
            except:
                print('Warning: Log dir not found and unable to create. Default to ./runs')
                log_dir = './runs'
                if not os.path.exists('./runs'):
                    os.mkdir('./runs')
        writer = SummaryWriter(log_dir)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.1)
        
        epoch_losses = []
        start = time.time()
        for e in range(start_from, start_from+epochs):
            self.model.train()
            epoch_loss = 0.
            for t, (sample, label) in enumerate(loader_train):  
                sample, label = self._input_augment_fn(sample, label)
                z = self.model(sample)                 
                optimizer.zero_grad()
                # if self.model_name == 'AEClassifier':
                #     for param in self.model.classifier.parameters():
                #         print(param.grad)
                if self.model_name == 'AE':
                    loss = self.loss(z, sample)
                else:
                    loss = self.loss(z, label)
                loss.backward()
                # if self.model_name == 'AEClassifier':
                #     for param in self.model.classifier.parameters():
                #         print(param.grad)

                optimizer.step()
                
                epoch_loss += loss.item()
                if writer:
                    # print(e*len(loader_train)+t)
                    writer.add_scalar("Loss/train", loss.item(), e*len(loader_train)+t)

                del label
                del z
                del loss
            # grads = []
            # for param in self.model.parameters():
            #     if param.grad != None:
            #         grads.append(torch.sum(param.grad))
            print("epoch {}:".format(e)) # iteration
            print('loss={}'.format(epoch_loss/len(loader_train)))
            print('weights={}'.format(sum_weights(self.model)))
            # print('gradients={}'.format(grads))
            epoch_losses.append(epoch_loss/len(loader_train))
            
            if self.model_name != "AE":
                print('acc={}'.format(self.score(loader_train)))
            print()
            
            # lr decay every lr_decay_nepoch epochs
            if e > 0 and e % lr_decay_nepoch == 0:
                print('Decay learning rate')
                scheduler.step()

            if save_every:
                # Save model save_every epoch
                if e > 0 and e % save_every == 0:
                    torch.save(self.model.state_dict(), f"{checkpoint_path}/epoch_{e}")
            
            # early stopping
            if len(epoch_losses) > 5 and np.all(np.absolute(np.diff(epoch_losses[-6:])) < early_stopping_eps):
                print(f'Losses change less than {early_stopping_eps} over last 5 epochs. Early stop')
                break
            if len(epoch_losses) > 10 and np.all(np.diff(epoch_losses[-11:]) > 0):
                print('Losses increase over last 10 epochs. Early stop')
                break   
        print('Training time', time.time()-start)
        # save final model
        now = datetime.datetime.now()
        timestamp = now.strftime("%y%m%d%H%M%S")
        torch.save(self.model.state_dict(), f"{checkpoint_path}/{timestamp}")
