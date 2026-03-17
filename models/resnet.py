from torchvision.models import resnet18


def get_resnet_model(num_classes: int = 10):
    return resnet18(num_classes=num_classes)
