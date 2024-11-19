from torchvision import transforms

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15, fill=1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"
            ),
            transforms.Normalize(
                mean=[0.853660523891449, 0.8536604642868042, 0.8536606431007385],
                std=[0.2529744505882263, 0.25297433137893677, 0.2529739737510681],
            ),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.853660523891449, 0.8536604642868042, 0.8536606431007385],
                std=[0.2529744505882263, 0.25297433137893677, 0.2529739737510681],
            ),
        ]
    ),
}
