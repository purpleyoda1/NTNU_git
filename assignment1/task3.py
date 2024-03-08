import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    
    #First need to make the output encoded the same way as the targets
    outputs = model.forward(X)
    max_indices = np.argmax(outputs, axis= 1)
    encoded_output = np.zeros_like(outputs)
    encoded_output[np.arange(outputs.shape[0]), max_indices] = 1

    #Then calc acc
    accuracy = np.sum(np.all(targets == encoded_output, axis= 1)) / targets.shape[0]

    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        
        logits = self.model.forward(X_batch)

        self.model.backward(X_batch, logits, Y_batch)

        self.model.w -= self.learning_rate * self.model.grad

        return cross_entropy_loss(Y_batch, logits)

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def plot_weights(weights, modelname, img_shape= (28, 28)):
    num_classes = weights.shape[1]
    weights = weights[:-1, :]

    #figsize= (num_classes*2, 2)
    fig, axes = plt.subplots(1, num_classes, figsize= (num_classes*2, 2))

    for i, ax in enumerate(axes):
        image_of_weights = weights[:, i].reshape(img_shape)
        ax.imshow(image_of_weights, cmap = 'gray')
        ax.axis('off')
    
    plt.savefig("task4b_softmax_weight_" + modelname + ".png")
    plt.show()


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    """ plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show() """

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model_lambda_1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model_lambda_1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg1, val_history_reg1 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    """ plot_weights(model.w, "model")
    plot_weights(model_lambda_1.w, "model_lamda_1")

    # Task 4c
    utils.plot_loss(val_history_reg1["accuracy"], "Lambda = 1")
    #lambda = 0.1
    model_lambda_01= SoftmaxModel(l2_reg_lambda= 0.1)
    trainer = SoftmaxTrainer(
        model_lambda_01, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    _, val_history_reg01 = trainer.train(num_epochs)
    utils.plot_loss(val_history_reg01["accuracy"], "Lambda = 0.1")

    #lambda = 0.01
    model_lambda_001 = SoftmaxModel(l2_reg_lambda= 0.01)
    trainer = SoftmaxTrainer(
        model_lambda_001, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    _, val_history_reg001 = trainer.train(num_epochs)
    utils.plot_loss(val_history_reg001["accuracy"], "Lambda = 0.01")

    #lambda = 0.001
    model_lambda_0001 = SoftmaxModel(l2_reg_lambda= 0.001)
    trainer = SoftmaxTrainer(
        model_lambda_0001, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    _, val_history_reg0001 = trainer.train(num_epochs)
    utils.plot_loss(val_history_reg0001["accuracy"], "Lambda = 0.001")
    plt.ylim([0.70, .93])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    models = [model_lambda_1, model_lambda_01, model_lambda_001, model_lambda_0001]
    l2_norms = []
    lambdas = [1, 0.1, 0.01, 0.001]

    for model in models:
        w = model.w
        w_norm = np.linalg.norm(w)
        l2_norms.append(w_norm)

    plt.plot(lambdas, l2_norms, marker= 'o', linestyle= '-')
    plt.xlabel('Lambda values')
    plt.ylabel('L2 norm of weights')
    plt.title('Weight norm by lambda values')
    plt.grid(True)
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show() """

if __name__ == "__main__":
    main()
