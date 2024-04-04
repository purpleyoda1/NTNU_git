import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import torchvision


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained= True)
        self.model.fc = nn.Linear(512, 10)

        img_size = 32
        num_filters = 32  # Set number of filters in first conv layer           #####   NUMBER OF FILTERS   #####
        self.num_classes = num_classes        

        #Freeze and modify parameters in layers
        for param in self.model.parameters():                                                                                     
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        batch_size = x.shape[0]

        #Check size
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.num_classes)
        assert x.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {x.shape}"
        return x


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()




def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")


if __name__ == "__main__":
    main()
