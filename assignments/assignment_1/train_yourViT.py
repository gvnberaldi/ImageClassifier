## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.class_model import DeepClassifier # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset

from dlvc.models.vit import VisionTransformer

from torch import nn

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR

from torchinfo import summary

import wandb

def train(config=None, num_epochs=30):

    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure 
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation
    
    with wandb.init(config=config, project = 'dlvc_ass_1_vit_sweep'):
        config = wandb.config

        #Data 

        augmentation_transform = v2.Compose([
            v2.ToImage(),
            # Randomly flip the image horizontally
            v2.RandomHorizontalFlip(),
            # Randomly rotate the image
            v2.RandomRotation(30),
            # Randomly crop and resize
            v2.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            # Convert the image to a PyTorch tensor
            v2.ToDtype(torch.float32, scale=True),
            # Normalize the image with mean and standard deviation
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    

        fdir = "data\\cifar-10-batches-py"

        train_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TRAINING, transform=transform)
        train_data.set_augmentation_transform(augmentation_transform=augmentation_transform, augment_probability=config.augmentation_ratio)
        val_data = CIFAR10Dataset(fdir=fdir, subset=Subset.VALIDATION, transform=transform) 
        test_data = CIFAR10Dataset(fdir=fdir, subset=Subset.TEST, transform=transform)
    
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Network

        network = VisionTransformer(
            img_size=32,
            patch_size=config.patch_size,
            in_chans=3,
            embed_dim=config.embed_dim,
            num_encoder_layers=config.num_encoder_layers,
            number_hidden_layers=config.number_hidden_layers,
            hidden_layer_depth = config.hidden_layer_depth,
            head_dim=config.head_dim,
            num_heads = config.num_heads,
            norm_layer=nn.LayerNorm,
            activation_function=nn.GELU,
            dropout=config.dropout,
            num_classes = 10,
            mlp_head_number_hidden_layers=config.mlp_head_number_hidden_layers,
            mlp_head_hidden_layers_depth=config.mlp_head_hidden_layers_depth
        )
        summary(network, input_size=(config.batch_size, 3, 32, 32))

        optimizer = AdamW(network.parameters(),lr=config.lr, amsgrad=True) if config.optimizer == 'AdamW' else SGD(network.parameters(),lr=config.lr, momentum = 0.9)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)
        val_frequency = 5

        model_save_dir = Path("saved_models\\vit")
        model_save_dir.mkdir(exist_ok=True)

        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=config.gamma)

        network = network.to(device)
    
        trainer = ImgClassificationTrainer(network, 
                        optimizer,
                        loss_fn,
                        lr_scheduler,
                        train_metric,
                        val_metric,
                        train_data,
                        val_data,
                        device,
                        num_epochs, 
                        model_save_dir,
                        batch_size=config.batch_size, # feel free to change
                        val_frequency = val_frequency)
        trainer.train()


if __name__ == "__main__":
    # Perform a parameter sweep using the provided functionality from WandB https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=ImLCIMOoIe5Y
    wandb.login(key="5a4726d6cfbe6bf6fa6cdab8143ed9b4f47db04d")
    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'validation-accuracy',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'optimizer': {
            'value': 'AdamW'
        },

        'scheduler' : {
            'value': 'ExponentialLR'
        },

        'lr':{
            'value': 0.001
        }, 

        'gamma':{
            'value': 0.95,
        },

        'patch_size': {
            'value': 4
        },

        'embed_dim': {
            'value': 32
        },

        'num_encoder_layers': {
            'value': 3
        },

        'number_hidden_layers': {
            'value': 2
        },

        'hidden_layer_depth': {
            'value': 1024
        },

        'head_dim': {
            'value': 32
        },

        'num_heads': {
            'value': 5
        },

        'dropout':{
            'value': 0
        },


        #'weight_decay':{
        #    'distribution': 'log_uniform_values',
        #    'min': 0.0001,
        #    'max': 0.1 
        #},

        'mlp_head_number_hidden_layers': {
            'value': 2
        },

        'mlp_head_hidden_layers_depth': {
            'value': 128
        },

        'batch_size': {
            'value': 512
        },

        'augmentation_ratio': {
            'value': 1
        }

    }

    config_final  = {
        'optimizer':  'AdamW'
        ,

        'scheduler' : 'ExponentialLR'
        ,

        'lr':0.001
        , 

        'gamma':0.95,
        

        'patch_size': 2
        ,

        'embed_dim': 256
        ,

        'num_encoder_layers': 3
        ,

        'number_hidden_layers': 1
        ,

        'hidden_layer_depth': 1024
        ,

        'head_dim': 32
        ,

        'num_heads': 5
        ,

        'dropout':0
        ,


        #'weight_decay':{
        #    'distribution': 'log_uniform_values',
        #    'min': 0.0001,
        #    'max': 0.1 
        #},

        'mlp_head_number_hidden_layers': 1
        ,

        'mlp_head_hidden_layers_depth': 128
        ,

        'batch_size': 256
        ,

        'augmentation_ratio': 1
        }

    #sweep_config['parameters'] = parameters_dict

    #sweep_id = wandb.sweep(sweep_config, project = 'dlvc_ass_1_vit_sweep')

    #wandb.agent(sweep_id, train, count=100)


    train(config_final, 120)