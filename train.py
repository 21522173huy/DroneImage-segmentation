
import torch
import numpy as np
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os
import random
from dataset import UAVDataset
from trainer import UAVSegmentationTrainer
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

def get_parser():
    # Create the parser
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default = 8)
    parser.add_argument('--epochs', type=int, default = 40)
    parser.add_argument('--eval_every', type=int, default = 2)
    parser.add_argument('--continue_epoch', type=int, default = )
    return parser

def main():
    # Create the parser
    parser = get_parser()
    args = parser.parse_args()
    
    # Building Dataset
    print('Loading Dataset and Dataloader')
    p = 0.5
    input_shape = 256

    train_transform = A.Compose([
              A.RandomResizedCrop(
                height=input_shape,
                width=input_shape,
                scale=(0.5, 1)
                ),
              A.RandomRotate90(p=p),
              A.HorizontalFlip(p=p),
              A.VerticalFlip(p=p),
              ToTensorV2()
        ])
    val_transform = A.Compose([A.Resize(height=input_shape, width=input_shape), ToTensorV2()])

    train_data = UAVDataset(args.train_file, transform=train_transform)
    val_data = UAVDataset(args.val_file, transform=val_transform)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)

    print(f'Length Train Dataset: {len(train_data)} | Length Val Dataset: {len(val_data)}')
    print(f'Length Train DataLoader: {len(train_dataloader)} | Length Val DataLoader: {len(val_dataloader)}')

    # Importing Model
    print(f'Loading Model: {args.model_name}')
    if args.model_name == 'unet3plus':
      from model.unet3plus import UNet3Plus
      model_instance = UNet3Plus(n_channels = 3, n_classes = len(train_data.get_label()))

    elif args.model_name == 'deeplab_v3':
      from model.deeplab_v3 import DeepLabv3
      model_instance = DeepLabv3()
      for p in model.deeplabv3.backbone.parameters():
        p.requires_grad = False

    elif args.model_name == 'sam':
      from model.deeplab_v3 import DeepLabv3
      model_instance = SegmentationVITSAM(embed_dim=768,
                                num_heads=12,
                                depth=12,
                                extract_layers=[3, 6, 9, 12],
                                encoder_global_attn_indexes=[2, 5, 8, 11],
                                drop_rate=0.1,
                                num_classes=len(train_data.get_label()),
      )
      
      for params in model.encoder.parameters():
          params.requires_grad = False
    elif args.model_name == 'unet':
      from model.unet import UNET
      model_instance = UNET(in_channels = 3, out_channels = len(train_data.get_label()))
    else :
      raise ValueError(f"The given model_name is invalid. Please check that model_name can be: unet3plus, deeplab_v3, sam, unet. Your given model name is {args.model_name}")
    
    pytorch_total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    print(f'Model Parameters: {pytorch_total_params}')

    # Training Config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_instance.parameters(), betas=(0.85, 0.95), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision = True
    
    # Training
    from trainer import UAVSegmentationTrainer
    trainer_instance = UAVSegmentationTrainer(model=model_instance,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    mixed_precision=True)
    
    trainer_instance.fit(epochs=args.epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            eval_every=args.eval_every,
            continue_epoch=args.continue_epoch)

if __name__ == "__main__":
    main()
