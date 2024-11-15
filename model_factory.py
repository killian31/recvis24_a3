"""Python file to instantite the model and the transform that goes with it."""

from torchvision.models import resnet18, resnet34

from data import data_transforms
from model import Net, ResNet


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
        elif self.model_name == "resnet18_pretrained":
            return resnet18(pretrained=True)
        elif self.model_name == "resnet34_pretrained":
            return resnet34(pretrained=True)
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if (
            self.model_name == "basic_cnn"
            or self.model_name == "resnet"
            or self.model_name == "resnet18_pretrained"
            or self.model_name == "resnet34_pretrained"
        ):
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
