import numpy as np
import utils
import typing

np.random.seed(1)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

def calculate_mu(num_samples):
    x = np.random.uniform(0, 1, num_samples)
    values = sigmoid(x)
    mu = np.mean(values)
    return mu

def sigmoid_improved(x, mu):
    return 1 / (1 + np.exp(-x - mu))
            

def pre_process_images(X: np.ndarray, X_train):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    deviation = np.std(X_train)
    mean = np.mean(X_train)
    X_normalized = (X-mean)/deviation

    #bias trick
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((X_normalized, ones))

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    C_n = -np.sum(targets * np.log(outputs), axis= 1)
    C = np.mean(C_n)
    #print(f"Loss: \n{C}\n\n")
    
    return C


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
        print_values: bool
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        self.print_values = print_values

        #Creating mu for improved sigmoid
        self.mu = calculate_mu(10000)
        print(f"mu is: {self.mu}")

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize everything
        self.ws = []
        self.momentum = []
        self.z = []
        self.outputs = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if (use_improved_weight_init):
                w = np.random.normal(0, np.sqrt(1/prev), (w_shape))
            else:
                w = np.random.uniform(-1, 1, (w_shape))
            self.ws.append(w)
            self.momentum.append(np.zeros(w.shape))
            prev = size

        
        self.grads = [np.zeros_like(w) for w in self.ws]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        self.outputs = [X]

        for i in range(len(self.ws) - 1):
            if self.print_values:
                print(f"(shape of ws: {self.ws[i].shape})")
            z = self.outputs[-1].dot(self.ws[i])
            if self.print_values:
                print(f"Before appending, shape of z is: {z.shape}")
            self.z.append(z)
            if (self.use_improved_sigmoid):
                if self.print_values:
                    print(f"Before sigmoid_improved, shape of z: {z.shape}")
                activation = sigmoid_improved(z, self.mu)
                if self.print_values:
                    print(f"After sigmoid_improved, shape of activation: {activation.shape}")
            else:
                activation = sigmoid(self.z[i])
            
            self.outputs.append(activation)
            
        z = self.z[-1].dot(self.ws[-1])
        self.z.append(z)
        activation = np.exp(self.z[-1])/np.sum(np.exp(self.z[-1]), axis= 1, keepdims= True)
        self.outputs.append(activation)

        if self.print_values:
            print(f"Shape of the outputs: {self.outputs[-1].shape}")

        return self.outputs[-1]

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        batch_size = X.shape[0]

        """ error = -(targets - outputs)
        self.grads[len(self.ws)-1] = np.dot(self.outputs[-2].T, error) / batch_size

        for i in reversed(range(len(self.ws) - 1)):
            error = sigmoid_derivative(self.outputs[i + 1]) * np.dot(error, self.ws[i + 1].T)
            self.grads[i] = np.dot(self.outputs[i].T, error) / batch_size """




        error =  outputs - targets
        self.grads[-1] = np.dot(self.outputs[-2].T, error) / batch_size
        error = np.dot(error, self.ws[-1].T)

        for i in reversed(range(len(self.ws) - 1)):
            if self.use_improved_sigmoid:
                error *= sigmoid_derivative(self.outputs[i + 1])  
            else:
                error *= self.outputs[i + 1] * (1 - self.outputs[i + 1]) 

            self.grads[i] = np.dot(self.outputs[i].T, error) / batch_size

            if i != 0: 
                error = np.dot(error, self.ws[i].T)


        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."


    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    zero_array = np.zeros((Y.shape[0], num_classes))
    for i in range (Y.shape[0]):
        zero_array[i][Y[i][0]] = 1

    return zero_array


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train, X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
