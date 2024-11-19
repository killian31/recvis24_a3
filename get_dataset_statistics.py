import argparse

import torch
from torchvision import datasets, transforms
from tqdm import tqdm


def calculate_mean_and_std(loader):
    """Calculate the mean and std of a dataset from a DataLoader."""
    channels_sum, channels_squared_sum, total_pixels = 0, 0, 0

    for data, _ in tqdm(loader):
        # Flatten the batch to [B * H * W, C]
        data = data.view(-1, data.size(1))
        channels_sum += data.sum(dim=0)
        channels_squared_sum += (data**2).sum(dim=0)
        total_pixels += data.size(0)

    mean = channels_sum / total_pixels  # Total mean per channel
    std = ((channels_squared_sum / total_pixels) - (mean**2)).sqrt()  # Std per channel

    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Calculate dataset statistics")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        help="Path to the dataset containing train_images folder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the DataLoader (default: 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader (default: 4)",
    )
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=f"{args.data}/train_images", transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Calculating mean and standard deviation...")
    mean, std = calculate_mean_and_std(train_loader)
    print(f"Dataset Statistics:\nMean: {mean.tolist()}\nStd: {std.tolist()}")


if __name__ == "__main__":
    main()
