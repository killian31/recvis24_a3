import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchinfo import summary
from torchvision import datasets

from model_factory import ModelFactory
from utils import WarmupScheduler


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        metavar="N",
        help="number of epochs to train (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2929,
        metavar="S",
        help="random seed (default: 2929)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Use a warmup scheduler",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=100,
        metavar="WI",
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Use a CosineAnnealing scheduler",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.00001,
        help="Minimum learning rate for CosineAnnealing scheduler",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the backbone of the model",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Use early stopping",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        metavar="ESP",
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        metavar="R",
        help="Path to a model to train from",
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
    warmup_scheduler: WarmupScheduler,
    val_loss: float,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
        warmup_scheduler (WarmupScheduler): Warmup scheduler
        scheduler (ReduceLROnPlateau): Plateau scheduler
        val_loss (float): Validation loss
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if not warmup_scheduler.is_warmup_done() and args.warmup:
            warmup_scheduler.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0  # Total samples

    with torch.no_grad():
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # Sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(output, target)
            validation_loss += loss.item() * data.size(0)
            # Get predictions
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    validation_loss /= total
    accuracy = 100.0 * correct / total
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            validation_loss,
            correct,
            total,
            accuracy,
        )
    )
    return validation_loss


def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms = ModelFactory(args.model_name).get_all()

    if args.resume_from is not None:
        map_location = "cuda" if use_cuda else "cpu"
        model.load_state_dict(torch.load(args.resume_from, map_location=map_location))

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    print(summary(model, input_size=(args.batch_size, 3, 224, 224), verbose=1))

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.data + "/train_images", transform=data_transforms["train"]
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.data + "/val_images", transform=data_transforms["val"]
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=1e-4,
    )

    warmup_scheduler = WarmupScheduler(optimizer, args.warmup_iters, args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # Loop over the epochs
    best_val_loss = 1e8
    val_loss = 1e8
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        # training loop
        train(
            model,
            optimizer,
            train_loader,
            use_cuda,
            epoch,
            args,
            warmup_scheduler,
            val_loss,
        )
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if args.scheduler:
            scheduler.step()
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        else:
            epochs_no_improve += 1
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
        if epochs_no_improve >= args.early_stopping_patience and args.early_stopping:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
