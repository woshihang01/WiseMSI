from torch import nn
from torchvision import models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(n_classes):
    model = models.resnet18(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, n_classes),
    )
    model = model.to(device)
    return model
