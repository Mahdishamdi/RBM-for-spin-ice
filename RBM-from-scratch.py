import numpy as np
import math
import os
from time import time
from typing import Tuple, Generator
import typer


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with sigmoid applied element-wise.
    """
    return 1 / (1 + np.exp(-x))


class RBM:
    """
    Restricted Boltzmann Machine (RBM) implementation.
    
    Attributes:
        n_vis (int): Number of visible units.
        n_hid (int): Number of hidden units.
        W (np.ndarray): Weight matrix connecting visible and hidden units.
        vbias (np.ndarray): Bias vector for visible units.
        hbias (np.ndarray): Bias vector for hidden units.
        W_grad (np.ndarray): Gradient for the weight matrix.
        vbias_grad (np.ndarray): Gradient for the visible bias.
        hbias_grad (np.ndarray): Gradient for the hidden bias.
        W_vel (np.ndarray): Momentum for weight updates.
        vbias_vel (np.ndarray): Momentum for visible bias updates.
        hbias_vel (np.ndarray): Momentum for hidden bias updates.
    """
    def __init__(self, n_vis: int, n_hid: int):
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W = 0.01 * np.random.randn(n_vis, n_hid)
        self.vbias = np.zeros(n_vis)
        self.hbias = np.zeros(n_hid)
        self.W_grad = np.zeros(self.W.shape)
        self.vbias_grad = np.zeros(n_vis)
        self.hbias_grad = np.zeros(n_hid)
        self.W_vel = np.zeros(self.W.shape)
        self.vbias_vel = np.zeros(n_vis)
        self.hbias_vel = np.zeros(n_hid)

    def h_given_v(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute hidden unit activations given visible units.

        Args:
            v (np.ndarray): Visible units.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Probabilities and sampled states of hidden units.
        """
        p = sigmoid(np.matmul(v, self.W) + self.hbias)
        return p, np.random.binomial(1, p=p)

    def v_given_h(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute visible unit activations given hidden units.

        Args:
            h (np.ndarray): Hidden units.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Probabilities and sampled states of visible units.
        """
        p = sigmoid(np.matmul(h, self.W.T) + self.vbias)
        return p, np.random.binomial(1, p=p)

    def compute_error_and_grads(self, batch: np.ndarray) -> float:
        """
        Compute reconstruction error and gradients for training.

        Args:
            batch (np.ndarray): Training batch.

        Returns:
            float: Reconstruction error.
        """
        b_size = batch.shape[0]
        v0 = batch.reshape(b_size, -1)

        ph0, h0 = self.h_given_v(v0)
        W_grad = np.matmul(v0.T, ph0)
        vbias_grad = np.sum(v0, axis=0)
        hbias_grad = np.sum(ph0, axis=0)

        pv1, v1 = self.v_given_h(h0)
        ph1, h1 = self.h_given_v(pv1)

        W_grad -= np.matmul(pv1.T, ph1)
        vbias_grad -= np.sum(pv1, axis=0)
        hbias_grad -= np.sum(ph1, axis=0)

        self.W_grad = W_grad / b_size
        self.hbias_grad = hbias_grad / b_size
        self.vbias_grad = vbias_grad / b_size

        recon_err = np.mean(np.sum((v0 - pv1) ** 2, axis=1))
        return recon_err

    def update_params(self, lr: float, momentum: float = 0.0):
        """
        Update model parameters using gradient descent with momentum.

        Args:
            lr (float): Learning rate.
            momentum (float): Momentum factor for updates.
        """
        self.W_vel = momentum * self.W_vel + lr * self.W_grad
        self.W += self.W_vel

        self.vbias_vel = momentum * self.vbias_vel + lr * self.vbias_grad
        self.vbias += self.vbias_vel

        self.hbias_vel = momentum * self.hbias_vel + lr * self.hbias_grad
        self.hbias += self.hbias_vel

    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data using the trained RBM.

        Args:
            v (np.ndarray): Input data.

        Returns:
            np.ndarray: Reconstructed data.
        """
        ph0, h0 = self.h_given_v(v)
        pv1, _ = self.v_given_h(ph0)
        return pv1


def main(file_path: str = '/home/mahdis/coded_6lat_T=3.5_600k.npy', epochs: int = 15000, lr: float = 0.0001, batch_size: int = 50, momentum: float = 0.9):
    """
    Main training loop for the RBM.
    """
    X_train = np.load(file_path)
    rbm = RBM(n_vis=36, n_hid=36)

    errors = []
    start_time = time()

    for epoch in range(1, epochs + 1):
        epoch_error = 0
        for batch in get_batches(X_train, batch_size):
            epoch_error += rbm.compute_error_and_grads(batch)
            rbm.update_params(lr, momentum)
        errors.append(epoch_error)
        print(f"Epoch {epoch} Error: {epoch_error:.4f} Time: {time() - start_time:.2f} s")

    end_time = time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Minimum Error: {min(errors):.4f}")


typer.run(main)
