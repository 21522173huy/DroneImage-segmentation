from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision.models.segmentation.fcn import FCNHead


class DeepLabv3(nn.Module):
    def __init__(self, outputchannels=8):
        super(DeepLabv3, self).__init__()
        self.deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                            progress=True)
        self.deeplabv3.classifier = DeepLabHead(2048, outputchannels)
        self.deeplabv3.aux_classifier = FCNHead(1024, outputchannels)
    

    def forward(self, pixel_values, labels):
        outputs = self.deeplabv3(pixel_values)

        loss=None
        if labels is not None:
            loss = self.compute_loss(outputs['out'], outputs['aux'], labels)
        
        
        return {
            'loss': loss,
            'logits': outputs['out'],
        }

    def inference(self, pixel_values):
        with torch.no_grad():
          outputs = self.deeplabv3(pixel_values)
        return outputs['out']
           
        
    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        if logits.shape[-1] != labels.shape[-1]:
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            upsampled_logits = logits


        if auxiliary_logits.shape[-1] != labels.shape[-1]:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            upsampled_auxiliary_logits = auxiliary_logits


        # compute weighted loss
        loss_fct = CrossEntropyLoss()
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        if auxiliary_logits is not None:
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += 0.4 * auxiliary_loss

        return loss
