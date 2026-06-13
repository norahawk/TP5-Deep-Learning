"""
layers.py — Shared building block for all exercises (1a, 1b, 2)
===============================================================
Contains:
  - Activation functions and their derivatives
  - Layer class: dense layer with forward/backward pass and Adam optimizer

Import this in every exercise file:
    from layers import Layer
"""

import numpy as np


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================
# Each function also exposes its derivative computed from the OUTPUT (not the
# pre-activation input). This is cheaper and numerically identical:
#   sigmoid'(z) = sigmoid(z)*(1-sigmoid(z)) = a*(1-a)
#   tanh'(z)    = 1 - tanh(z)²             = 1 - a²
# =============================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def tanh_act(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1.0 - a ** 2


# =============================================================================
# LAYER CLASS
# =============================================================================

class Layer:
    """
    One fully-connected layer with a chosen activation and Adam optimizer.

    Parameters
    ----------
    n_in  : number of input features
    n_out : number of neurons
    activation : 'tanh' | 'sigmoid' | 'linear'
    seed  : random seed (use different seeds per layer for independent init)

    Usage
    -----
    layer = Layer(35, 16, activation='tanh', seed=0)
    out   = layer.forward(x)          # forward pass
    d_in, dW, db = layer.backward(d)  # backward pass
    layer.update(dW, db, lr=0.005)    # Adam weight update
    """

    def __init__(self, n_in, n_out, activation='tanh', seed=0):
        rng = np.random.default_rng(seed)

        # He initialization: keeps activation variance stable across layers.
        # Scale = sqrt(2/n_in). Works well for tanh and sigmoid.
        scale = np.sqrt(2.0 / n_in)
        self.W = rng.normal(0, scale, (n_in, n_out))
        self.b = np.zeros((1, n_out))

        self.activation = activation

        # Adam state — one pair (m, v) per parameter matrix
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0   # step counter for bias correction

        # Cache filled during forward pass, used in backward pass
        self.input  = None
        self.output = None

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x : (batch_size, n_in)
        Returns output of shape (batch_size, n_out)
        """
        self.input = x
        pre_act = x @ self.W + self.b          # linear: z = xW + b

        if self.activation == 'tanh':
            self.output = tanh_act(pre_act)
        elif self.activation == 'sigmoid':
            self.output = sigmoid(pre_act)
        else:                                  # 'linear' — identity
            self.output = pre_act

        return self.output

    # ------------------------------------------------------------------
    def backward(self, d_out):
        """
        d_out : gradient of loss w.r.t. this layer's output, shape (batch, n_out)

        Returns
        -------
        d_in : gradient w.r.t. input  (pass to the previous layer)
        dW   : gradient w.r.t. W
        db   : gradient w.r.t. b
        """
        # Chain rule: multiply incoming gradient by activation derivative
        if self.activation == 'tanh':
            d_pre = d_out * tanh_deriv(self.output)
        elif self.activation == 'sigmoid':
            d_pre = d_out * sigmoid_deriv(self.output)
        else:
            d_pre = d_out   # linear: derivative = 1

        dW   = self.input.T @ d_pre
        db   = d_pre.sum(axis=0, keepdims=True)
        d_in = d_pre @ self.W.T

        return d_in, dW, db

    # ------------------------------------------------------------------
    def update(self, dW, db, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam optimizer update.

        beta1 = 0.9   : momentum decay (smooths gradient direction)
        beta2 = 0.999 : RMSProp decay (scales step size per parameter)
        eps           : numerical stability constant
        """
        self.t += 1

        # Update biased moment estimates
        self.mW = beta1 * self.mW + (1 - beta1) * dW
        self.vW = beta2 * self.vW + (1 - beta2) * dW**2
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * db**2

        # Bias correction (compensates for zero-initialization)
        mW_hat = self.mW / (1 - beta1**self.t)
        vW_hat = self.vW / (1 - beta2**self.t)
        mb_hat = self.mb / (1 - beta1**self.t)
        vb_hat = self.vb / (1 - beta2**self.t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
