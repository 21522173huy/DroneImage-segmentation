
# DroneImage-segmentation
## Description
- In this repo, we did implement Instance Segmentation task on `UAV Drone` dataset. Through the project, our desire is understanding deeply Instance Segmentation's operation, as well as exploring its practical applications in the field of computer vision
- Dataset Information: https://uavid.nl/
- We did implement 4 segmentation model: 
    - `UNET`
    - `UNET+++`
    - `Deeplab_V3 `
    - `SAM (Segment Anything)`
- Especially, We followed `Unet+++` source code from repo `https://github.com/avBuffer/UNet3plus_pth`

## Introduction
- Unmanned Aerial Vehicles (UAVs), also known as drones, are revolutionizing data collection and analysis across various fields. One of the significant advancements in UAV technology is the ability to perform semantic segmentation, which provides detailed and optimized data information.
- Some notable benefits include urban planning and development, mapping and surveying, and regional security monitoring
  
## Usage
```
git clone https://github.com/21522173huy/DroneImage-segmentation
cd DroneImage-segmentation
```
- Before run `train.py` and `inference.py` script, please download, unzip and locate dataset into `DroneImage-segmentation` dicrectory
- For example: `uavid_dataset`

You can choose from the following model options for the `model_name` parameter:

- `unet3plus`
- `unet`
- `deeplab_v3`
- `sam`

## Train Script
```
python train.py \
--train_file uavid_dataset/uavid_train \
--val_file uavid_dataset/uavid_val \
--model_name unet3plus \
--epochs 40
```

## Inference Script
- After running train script, you will obtain checkpoint folder followed by the model_name, `deeplab_v3_checkpoints` for instance.
```
python inference.py \
--infer_file uavid_dataset/uavid_val \
--model_name deeplab_v3 \
--checkpoint_path deeplab_v3_checkpoints/checkpoint_40.pth
```

## Conclusion
- Semantic segmentation on UAV drones is very useful and has powerful and practical applications, such as surveying urban locations.
- The models we use in this research have shown noticeable performance. Although the results are not yet satisfactory, they are still very promising.
- Since there are still limitations in the size of our dataset, we will obtain and train our models on other datasets that are greater in terms of quantity and quality in the future

## Citation
```
@article{LYU2020108,
	author = "Ye Lyu and George Vosselman and Gui-Song Xia and Alper Yilmaz and Michael Ying Yang",
	title = "UAVid: A semantic segmentation dataset for UAV imagery",
	journal = "ISPRS Journal of Photogrammetry and Remote Sensing",
	volume = "165",
	pages = "108 - 119",
	year = "2020",
	issn = "0924-2716",
	doi = "https://doi.org/10.1016/j.isprsjprs.2020.05.009",
	url = "http://www.sciencedirect.com/science/article/pii/S0924271620301295",
}
```

```
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical image computing and computer-assisted intervention--MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

```
@inproceedings{huang2020unet,
  title={Unet 3+: A full-scale connected unet for medical image segmentation},
  author={Huang, Huimin and Lin, Lanfen and Tong, Ruofeng and Hu, Hongjie and Zhang, Qiaowei and Iwamoto, Yutaro and Han, Xianhua and Chen, Yen-Wei and Wu, Jian},
  booktitle={ICASSP 2020-2020 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={1055--1059},
  year={2020},
  organization={IEEE}
}
```

```
@inproceedings{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4015--4026},
  year={2023}
}
```

```
@article{chen2017rethinking,
  title={Rethinking atrous convolution for semantic image segmentation. arXiv},
  author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal={arXiv preprint arXiv:1706.05587},
  volume={5},
  year={2017}
}
```





