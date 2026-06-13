"""
autoencoder.py — Base Autoencoder class, shared by exercises 1a and 1b
=======================================================================
The Autoencoder stacks Layer objects into an encoder and decoder.
Exercise 1b (Denoising) imports this class directly and only changes
what goes into the network as input during training — the architecture
and training loop stay the same.

Usage
-----
from autoencoder import Autoencoder
from font_data import X

ae = Autoencoder(input_dim=35, encoder_dims=[16], latent_dim=2,
                 decoder_dims=[16], seed=42)
losses = ae.train(X, epochs=30000, lr=0.005)
Z      = ae.encode(X)   # (32, 2) — latent coordinates
X_hat  = ae.forward(X)  # (32, 35) — reconstructions
"""

import numpy as np
from layers import Layer


class Autoencoder:
    """
    Symmetric encoder-decoder network with a configurable bottleneck.

    Architecture
    ------------
    Input(input_dim)
      → [encoder_dims hidden layers, tanh]
      → Latent(latent_dim, linear)        ← the 2D bottleneck
      → [decoder_dims hidden layers, tanh]
      → Output(input_dim, sigmoid)

    The latent layer uses a linear (identity) activation so the 2D space
    is unconstrained — the network can place points anywhere in the plane.
    All hidden layers use tanh (zero-centered, avoids all-positive gradients).
    The output layer uses sigmoid to map reconstructed values into (0,1),
    matching binary pixel targets.

    Loss: Binary Cross-Entropy — penalises confident wrong predictions
    strongly, which is appropriate for binary (0/1) pixel data.

    Optimiser: Adam — adapts the learning rate per parameter using
    momentum and RMSProp, converges much faster than plain SGD for
    bottleneck architectures where encoder and decoder gradients differ.
    """

    def __init__(self, input_dim, encoder_dims, latent_dim, decoder_dims, seed=42):
        """
        Parameters
        ----------
        input_dim    : int — number of input/output features (35 for font data)
        encoder_dims : list[int] — hidden layer sizes in the encoder
                       e.g. [16] → one hidden layer: input_dim → 16 → latent_dim
        latent_dim   : int — size of the bottleneck (2 for this exercise)
        decoder_dims : list[int] — hidden layer sizes in the decoder
                       should mirror encoder_dims for a symmetric architecture
        seed         : int — base random seed for reproducibility
        """
        self.layers = []

        # --- Encoder hidden layers ---
        prev = input_dim
        for i, h in enumerate(encoder_dims):
            self.layers.append(Layer(prev, h, activation='tanh', seed=seed + i))
            prev = h

        # --- Latent layer (linear — no squashing at the bottleneck) ---
        self.layers.append(Layer(prev, latent_dim, activation='linear', seed=seed + 100))
        self.latent_idx = len(self.layers) - 1
        prev = latent_dim

        # --- Decoder hidden layers ---
        for i, h in enumerate(decoder_dims):
            self.layers.append(Layer(prev, h, activation='tanh', seed=seed + 200 + i))
            prev = h

        # --- Output layer: sigmoid maps to (0,1) for binary pixels ---
        self.layers.append(Layer(prev, input_dim, activation='sigmoid', seed=seed + 300))

    # ------------------------------------------------------------------
    def forward(self, x):
        """Full encoder → decoder pass. x: (batch, input_dim)."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def encode(self, x):
        """Encoder only. Returns latent vectors z of shape (batch, latent_dim)."""
        out = x
        for layer in self.layers[:self.latent_idx + 1]:
            out = layer.forward(out)
        return out

    def decode(self, z):
        """Decoder only. z: (batch, latent_dim) → (batch, input_dim)."""
        out = z
        for layer in self.layers[self.latent_idx + 1:]:
            out = layer.forward(out)
        return out

    # ------------------------------------------------------------------
    def bce_loss(self, x_hat, x_target, eps=1e-8):
        """
        Binary Cross-Entropy between reconstruction x_hat and target x_target.
        BCE = -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )
        """
        return -np.mean(
            x_target * np.log(x_hat + eps) +
            (1 - x_target) * np.log(1 - x_hat + eps)
        )

    def bce_gradient(self, x_hat, x_target, eps=1e-8):
        """Gradient of BCE w.r.t. x_hat (the network output)."""
        n = x_target.shape[0]
        return (-(x_target / (x_hat + eps)) + (1 - x_target) / (1 - x_hat + eps)) / n

    # ------------------------------------------------------------------
    def train(self, X_input, X_target=None, epochs=30000, lr=0.005, print_every=5000):
        """
        Training loop.

        Parameters
        ----------
        X_input  : (batch, input_dim) — what is fed into the encoder.
                   For the basic autoencoder (1a) this equals X_target.
                   For the denoising autoencoder (1b) this is the corrupted
                   version of X_target.
        X_target : (batch, input_dim) — the reconstruction target.
                   Defaults to X_input (standard autoencoder behaviour).
        epochs   : number of training iterations
        lr       : Adam learning rate
        print_every : print progress every N epochs

        Returns
        -------
        losses : list of BCE loss values, one per epoch
        """
        if X_target is None:
            X_target = X_input   # standard autoencoder: reconstruct the input

        losses = []
        for epoch in range(1, epochs + 1):
            # Forward pass
            x_hat = self.forward(X_input)

            # Loss
            loss = self.bce_loss(x_hat, X_target)
            losses.append(loss)

            # Backward pass — traverse layers in reverse order
            grad = self.bce_gradient(x_hat, X_target)
            for layer in reversed(self.layers):
                grad, dW, db = layer.backward(grad)
                layer.update(dW, db, lr)

            if epoch % print_every == 0:
                errors = self.pixel_errors(X_input, X_target)
                print(f"  Epoch {epoch:6d} | Loss: {loss:.6f} | Max wrong pixels: {errors.max()}")

        return losses

    # ------------------------------------------------------------------
    def pixel_errors(self, X_input, X_target=None):
        """
        Number of wrongly reconstructed pixels per sample.

        For the basic autoencoder: X_target = X_input (same).
        For the denoising autoencoder: X_target is the clean original,
        X_input is the noisy version — we measure error against the clean.
        """
        if X_target is None:
            X_target = X_input
        x_hat = self.forward(X_input)
        x_bin = (x_hat >= 0.5).astype(int)
        return np.sum(np.abs(x_bin - X_target.astype(int)), axis=1)
