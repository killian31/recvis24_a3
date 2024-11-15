"""Python file to instantite the model and the transform that goes with it."""

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

from data import data_transforms
from model import CustomCNN, Net, ResNet


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet":
            return ResNet()
        elif self.model_name == "custom_cnn":
            return CustomCNN()
        elif self.model_name == "resnet18_pretrained":
            model = resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 500)
            return model
        elif self.model_name == "resnet34_pretrained":
            model = resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 500)
            return model
        elif self.model_name == "resnet50_pretrained":
            model = resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 500)
            return model
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name in [
            "basic_cnn",
            "resnet",
            "custom_cnn",
            "resnet18_pretrained",
            "resnet34_pretrained",
            "resnet50_pretrained",
        ]:
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
