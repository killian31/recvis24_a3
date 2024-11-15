"""Python file to instantite the model and the transform that goes with it."""

import timm

from data import data_transforms
from model import CustomCNN, Net, ResNet

num_classes = 500


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
        else:
            try:
                return timm.create_model(
                    self.model_name, pretrained=True, num_classes=num_classes
                )
            except:
                raise ValueError(
                    f"Model {self.model_name} not found in timm models or custom models"
                )

    def init_transform(self):
        return data_transforms

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
