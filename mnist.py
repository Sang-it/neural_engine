import gzip
import os

import numpy as np
from tqdm import trange

import optim
from tensor import Tensor


def train(
    model,
    X_train,
    Y_train,
    optim,
    steps,
    BS=128,
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    transform=lambda x: x,
    target_transform=lambda x: x,
    noloss=False,
):
    Tensor.training = True
    losses, accuracies = [], []

    for _ in (tr := trange(steps)):
        # Randomly sample a batch of data
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        x = Tensor(transform(X_train[samp]), requires_grad=False)
        y = Tensor(target_transform(Y_train[samp]))

        # Forward pass
        out = model.forward(x) if hasattr(model, "forward") else model(x)

        # Compute loss and backpropagate
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        if noloss:
            del loss
        optim.step()

        # Calculate accuracy
        if not noloss:
            cat = np.argmax(out.numpy(), axis=-1)
            accuracy = (cat == y.numpy()).mean()

            loss_value = loss.detach().numpy()
            losses.append(loss_value)
            accuracies.append(accuracy)
            tr.set_description(f"Loss: {loss_value:.2f}, Accuracy: {accuracy:.2f}")

    return losses, accuracies


def evaluate(
    model,
    X_test,
    Y_test,
    num_classes=None,
    BS=128,
    return_predict=False,
    transform=lambda x: x,
    target_transform=lambda y: y,
):
    Tensor.training = False

    def numpy_eval(Y_test, num_classes):
        Y_test_preds_out = np.zeros(list(Y_test.shape) + [num_classes])

        # Iterate through test set in batches
        for i in trange((len(Y_test) - 1) // BS + 1, desc="Evaluating"):
            x = Tensor(transform(X_test[i * BS : (i + 1) * BS]))
            out = model.forward(x) if hasattr(model, "forward") else model(x)
            Y_test_preds_out[i * BS : (i + 1) * BS] = out.numpy()

        # Calculate accuracy
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        Y_test = target_transform(Y_test)
        return (Y_test == Y_test_preds).mean(), Y_test_preds

    if num_classes is None:
        num_classes = Y_test.max().astype(int) + 1

    # Evaluate accuracy and predictions
    acc, Y_test_pred = numpy_eval(Y_test, num_classes)
    print(f"Test set accuracy: {acc:.4f}")

    return (acc, Y_test_pred) if return_predict else acc


def fetch_mnist():
    def _parse(file):
        return np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()

    BASE = os.path.dirname(__file__) + "/extra/datasets"

    # Load MNIST dataset
    X_train = (
        _parse(BASE + "/mnist/train-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = _parse(BASE + "/mnist/train-labels-idx1-ubyte.gz")[8:]

    X_test = (
        _parse(BASE + "/mnist/t10k-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = _parse(BASE + "/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    return X_train, Y_train, X_test, Y_test


class TinyConvNet:
    def __init__(self):
        # Initialize network layers
        conv = 3
        inter_chan, out_chan = 8, 16
        self.c1 = Tensor.scaled_uniform(inter_chan, 1, conv, conv)
        self.c2 = Tensor.scaled_uniform(out_chan, inter_chan, conv, conv)
        self.l1 = Tensor.scaled_uniform(out_chan * 5 * 5, 10)

    def forward(self, x: Tensor):
        # Define forward pass through the network
        x = x.reshape(shape=(-1, 1, 28, 28))  # Reshape input for conv layers
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1).log_softmax()


if __name__ == "__main__":
    # Fetch MNIST data
    X_train, Y_train, X_test, Y_test = fetch_mnist()

    # Set random seed for reproducibility
    np.random.seed(1337)

    # Initialize model and optimizer
    model = TinyConvNet()
    optimizer = optim.Adam([model.c1, model.c2, model.l1], lr=0.001)

    # Train model
    train(model, X_train, Y_train, optimizer, steps=100)

    # Evaluate model
    evaluate(model, X_test, Y_test)
