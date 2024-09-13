import torch
import numpy as np
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os
import random
from dataset import UAVDataset
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
import evaluate
mean_iou = evaluate.load("mean_iou")

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
    parser.add_argument('--infer_file', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--batch_size', type=int, default = 8)
    return parser

def compute_metrics(predicted, ground_truth):
  # Compute Accuracy and mIoU
  results = mean_iou.compute(predictions=predicted, references=ground_truth, num_labels=8, ignore_index = None)
  mIoU = results['mean_iou']
  accuracy = results['overall_accuracy']

  # Compute F1
  y_pred_np = predicted.squeeze().detach().cpu().numpy()
  y_true_np = ground_truth.squeeze().detach().cpu().numpy()
  f1 = f1_score(y_true_np.flatten(), y_pred_np.flatten(), average = 'macro')

  return np.array([mIoU, accuracy, f1])

def infer(model, dataloader, compute_metrics, device = 'cpu', deeplab = False):
  total_metrics = 0
  model.eval(), model.to(device)

  for i, samples in enumerate(tqdm(dataloader)):
    if deeplab == True : predicted = model(samples['image'].to(device), labels = None)['logits'].argmax(dim=1)
    else : predicted = model(samples['image'].to(device)).argmax(dim=1)

    ground_truth = samples['label'].to(device)

    total_metrics += compute_metrics(predicted, ground_truth)

  total_metrics = total_metrics/len(dataloader)

  return {
      'mIoU' : total_metrics[0],
      'accuracy' : total_metrics[1],
      'f1_score' : total_metrics[2]
  }
    
def main():
    # Create the parser
    parser = get_parser()
    args = parser.parse_args()

    # Building Dataset
    print('Loading Dataset and Dataloader')
    p = 0.5
    input_shape = 256

    test_transform = A.Compose([A.Resize(height=input_shape, width=input_shape), ToTensorV2()])
    test_data = UAVDataset(args.infer_file, transform=test_transform)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    print(f'Length Train DataLoader: {len(test_dataloader)} | Length Val DataLoader: {len(test_dataloader)}')

    # Importing Model
    print(f'Loading Model: {args.model_name}')
    if args.model_name == 'unet3plus':
      from model.unet3plus import UNet3Plus
      model_instance = UNet3Plus(n_channels = 3, n_classes = len(test_data.get_label()))

    elif args.model_name == 'deeplab_v3':
      from model.deeplab_v3 import DeepLabv3
      model_instance = DeepLabv3()
      for p in model_instance.deeplabv3.backbone.parameters():
        p.requires_grad = False

    elif args.model_name == 'sam':
      from model.sam import SegmentationVITSAM
      model_instance = SegmentationVITSAM(embed_dim=768,
                                num_heads=12,
                                depth=12,
                                extract_layers=[3, 6, 9, 12],
                                encoder_global_attn_indexes=[2, 5, 8, 11],
                                drop_rate=0.1,
                                num_classes=len(test_data.get_label()),
      )

      for params in model_instance.encoder.parameters():
          params.requires_grad = False
    elif args.model_name == 'unet':
      from model.unet import UNET
      model_instance = UNET(in_channels = 3, out_channels = len(test_data.get_label()))
    else :
      raise ValueError(f"The given model_name is invalid. Please check that model_name can be: unet3plus, deeplab_v3, sam, unet. Your given model name is {args.model_name}")

    pytorch_total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    print(f'Model Parameters: {pytorch_total_params}')

    # Loading Pretrained-Weight
    checkpoint = torch.load(args.checkpoint_path)
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    print(f'All keys matched successfully')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cm = compute_metrics
    if args.model_name == 'deeplab_v3':
      metrics = infer(model_instance, test_dataloader, cm , device, deeplab = True)
    else:
      metrics = infer(model_instance, test_dataloader, cm, device)

    print(f'Result: \n', metrics)

if __name__ == "__main__":
    main()
