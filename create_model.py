import torch
import torchvision

from torch import nn

def create_effnetb2_model(num_classes:int=3, seed:int=42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for layer in [6,7]:
        for param in model.features[layer].parameters():
            param.requires_grad = True

    for param in model.features[8].parameters():
        param.requires_grad = True

    torch.manual_seed(seed)
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408,out_features=num_classes),
    )

    return model, transforms
