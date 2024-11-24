"""Python file to instantite the model and the transform that goes with it."""

import timm
import torch.nn as nn

from data import data_transforms, dino_transforms
from model import CustomCNN, ResNet

num_classes = 500


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "resnet":
            return ResNet()
        elif self.model_name == "custom_cnn":
            return CustomCNN()
        elif "efficientnet" in self.model_name:
            try:
                model = timm.create_model(
                    self.model_name, pretrained=True, num_classes=num_classes
                )
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(model.classifier.in_features, num_classes),
                )
                return model
            except Exception as e:
                raise ValueError(
                    f"Model {self.model_name} not found in timm models or custom models. Error: {e}"
                )
        elif "vit" in self.model_name or "deit" in self.model_name:
            try:
                model = timm.create_model(
                    self.model_name, pretrained=True, num_classes=num_classes
                )
                return model
            except Exception as e:
                raise ValueError(
                    f"Model {self.model_name} not found in timm models or custom models. Error: {e}"
                )
        else:
            raise ValueError(
                f"Model {self.model_name} not found in timm models or custom models."
            )

    def init_transform(self):
        if "dinov2" not in self.model_name:
            return data_transforms
        else:
            return dino_transforms

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
