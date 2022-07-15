from torch import nn
from torchvision import models
import torch
from pytorch_pretrained_vit import ViT
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_type, n_classes):
    if model_type == "resnet18":
        model = models.resnet18(pretrained=True)
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, n_classes)
        model = model.to(device)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, n_classes)
        model = model.to(device)
    elif model_type == "vit":
        model = ViT('B_32_imagenet1k', pretrained=True)
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, n_classes)
        model = model.to(device)
    elif model_type == 'efficient':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        fc_in_features = model._fc.in_features
        model._fc = nn.Linear(fc_in_features, n_classes)
        model = model.to(device)
    else:
        raise Exception("Invalid model name, exiting...")
    return model
