"""
exercise_1a.py — TP5 Exercise 1a: Basic Autoencoder
=====================================================
Depends on: layers.py, font_data.py, autoencoder.py

Items covered:
  1) Encoder/Decoder architecture with 2D latent space
  2) Architecture comparison and optimisation study
  3) 2D latent space plot
  4) New character generation from unseen latent points

Run:
    python exercise_1a.py
"""

import numpy as np
import matplotlib.pyplot as plt

from font_data import X, CHAR_LABELS        # 32×35 binary dataset
from autoencoder import Autoencoder         # shared encoder-decoder class


# =============================================================================
# ITEM 1 & 2 — Architecture and optimisation study
# =============================================================================
#
# Architecture A: 35 → 16 → 2 → 16 → 35  (chosen as the best)
# Architecture B: 35 → 25 → 10 → 2 → 10 → 25 → 35
#
# Key findings:
#   - Arch A is simpler and converges stably. With only 32 training samples,
#     a smaller model generalises the compression task better.
#   - Arch B has more parameters and converges faster initially, but Adam
#     overshoots around epoch 20-25k, causing a loss spike. More parameters
#     ≠ better performance when the dataset is tiny.
#   - Full-batch training (all 32 samples at once) is appropriate here.
#     Mini-batches would add noise that slows convergence with so few samples.
#   - lr=0.005 with Adam works well for Arch A.
# =============================================================================

print("="*60)
print("Architecture A: 35 → 16 → 2 → 16 → 35")
print("="*60)
ae_A = Autoencoder(input_dim=35, encoder_dims=[16], latent_dim=2,
                   decoder_dims=[16], seed=42)
losses_A = ae_A.train(X, epochs=30000, lr=0.005, print_every=5000)
errors_A = ae_A.pixel_errors(X)
print(f"\nArch A — Max wrong pixels: {errors_A.max()}  |  Mean: {errors_A.mean():.2f}")

print("\n" + "="*60)
print("Architecture B: 35 → 25 → 10 → 2 → 10 → 25 → 35")
print("="*60)
ae_B = Autoencoder(input_dim=35, encoder_dims=[25, 10], latent_dim=2,
                   decoder_dims=[10, 25], seed=7)
losses_B = ae_B.train(X, epochs=50000, lr=0.005, print_every=5000)
errors_B = ae_B.pixel_errors(X)
print(f"\nArch B — Max wrong pixels: {errors_B.max()}  |  Mean: {errors_B.mean():.2f}")

# Select best model (prefer A if it meets the goal)
if errors_A.max() <= 1:
    best_ae, best_errors, best_name = ae_A, errors_A, "A (35→16→2→16→35)"
elif errors_B.max() <= 1:
    best_ae, best_errors, best_name = ae_B, errors_B, "B (35→25→10→2→10→25→35)"
else:
    print("\nNeither model met ≤1 pixel. Continuing Arch B with lr=0.001...")
    ae_B.train(X, epochs=50000, lr=0.001, print_every=10000)
    errors_B = ae_B.pixel_errors(X)
    best_ae, best_errors, best_name = ae_B, errors_B, "B extended"

print(f"\n✓ Best model: {best_name}")
print(f"  Per-character errors: {best_errors.tolist()}")
print(f"  Goal (≤1 pixel) achieved: {best_errors.max() <= 1}")


# =============================================================================
# PLOT 1 — Loss curves (architecture comparison)
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(losses_A, label="Arch A: 35→16→2→16→35", alpha=0.85)
ax.plot(losses_B, label="Arch B: 35→25→10→2→10→25→35", alpha=0.85)
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("1a — Training Loss: Architecture Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("1a_loss_curve.png", dpi=150)
plt.close()
print("\nSaved: 1a_loss_curve.png")


# =============================================================================
# PLOT 2 — Reconstructions vs Originals
# =============================================================================

x_hat    = best_ae.forward(X)
x_bin    = (x_hat >= 0.5).astype(int)

fig, axes = plt.subplots(4, 16, figsize=(22, 7))
fig.suptitle(f"1a — Original (top) vs Reconstructed (bottom) — {best_name}", fontsize=11)

for i in range(32):
    col      = i % 16
    row_orig = (i // 16) * 2
    row_rec  = row_orig + 1

    axes[row_orig][col].imshow(X[i].reshape(7, 5), cmap='binary', vmin=0, vmax=1)
    axes[row_orig][col].set_title(f"'{CHAR_LABELS[i]}'", fontsize=7)
    axes[row_orig][col].axis('off')

    color = 'green' if best_errors[i] <= 1 else 'red'
    axes[row_rec][col].imshow(x_bin[i].reshape(7, 5), cmap='binary', vmin=0, vmax=1)
    axes[row_rec][col].set_title(f"err={best_errors[i]}", fontsize=7, color=color)
    axes[row_rec][col].axis('off')

plt.tight_layout()
plt.savefig("1a_reconstructions.png", dpi=150)
plt.close()
print("Saved: 1a_reconstructions.png")


# =============================================================================
# ITEM 3 — 2D Latent Space Plot
# =============================================================================
#
# Each of the 32 characters gets encoded to a (z1, z2) point.
# We expect:
#   - Similar-looking characters cluster together
#   - No two characters collapse to the same point
#   - The space is continuous: interpolating between two points gives
#     a character that blends features of both

Z = best_ae.encode(X)   # shape (32, 2)

fig, ax = plt.subplots(figsize=(11, 9))
colors = plt.cm.tab20(np.linspace(0, 1, 32))

for i, (z, label) in enumerate(zip(Z, CHAR_LABELS)):
    ax.scatter(z[0], z[1], color=colors[i], s=130, zorder=5,
               edgecolors='k', linewidths=0.4)
    ax.annotate(f"'{label}'", (z[0], z[1]),
                textcoords="offset points", xytext=(7, 4),
                fontsize=9, color=colors[i], fontweight='bold')

ax.set_xlabel("Latent dimension z₁", fontsize=12)
ax.set_ylabel("Latent dimension z₂", fontsize=12)
ax.set_title("1a — Item 3: All 32 Characters in the 2D Latent Space", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("1a_latent_space.png", dpi=150)
plt.close()
print("Saved: 1a_latent_space.png")


# =============================================================================
# ITEM 4 — Generate a New Character
# =============================================================================
#
# The decoder is a continuous function from R² → [0,1]^35.
# Any point NOT in the 32 training positions decodes to a new character.
#
# Method A — Interpolation: midpoint between 'a' (idx 1) and 'o' (idx 15).
#   The result blends features of both characters.
#
# Method B — Grid: scan a grid over the latent space and decode each point.
#   Every cell not coinciding with a training point is a new character.
#   The smooth transitions across the grid confirm the network has learned
#   a continuous representation, not just memorised 32 lookup entries.

print("\n" + "="*60)
print("Item 4 — Generating new characters")
print("="*60)

idx1, idx2 = 1, 15   # 'a' and 'o'
z_interp   = (Z[idx1] + Z[idx2]) / 2.0
z_centroid = Z.mean(axis=0)

char_interp  = best_ae.decode(z_interp.reshape(1, -1))
char_centroid = best_ae.decode(z_centroid.reshape(1, -1))

fig, axes = plt.subplots(1, 4, figsize=(10, 3))

axes[0].imshow(X[idx1].reshape(7, 5), cmap='binary')
axes[0].set_title(f"Original '{CHAR_LABELS[idx1]}'\nz=({Z[idx1,0]:.2f}, {Z[idx1,1]:.2f})")
axes[0].axis('off')

axes[1].imshow(X[idx2].reshape(7, 5), cmap='binary')
axes[1].set_title(f"Original '{CHAR_LABELS[idx2]}'\nz=({Z[idx2,0]:.2f}, {Z[idx2,1]:.2f})")
axes[1].axis('off')

axes[2].imshow((char_interp >= 0.5).astype(int).reshape(7, 5), cmap='binary')
axes[2].set_title(f"Midpoint (NEW)\nz=({z_interp[0]:.2f}, {z_interp[1]:.2f})")
axes[2].axis('off')

axes[3].imshow((char_centroid >= 0.5).astype(int).reshape(7, 5), cmap='binary')
axes[3].set_title(f"Centroid (NEW)\nz=({z_centroid[0]:.2f}, {z_centroid[1]:.2f})")
axes[3].axis('off')

fig.suptitle("1a — Item 4: New Characters via Latent Space Interpolation", fontsize=11)
plt.tight_layout()
plt.savefig("1a_new_characters.png", dpi=150)
plt.close()
print("Saved: 1a_new_characters.png")

# Grid exploration
z1_range = np.linspace(Z[:, 0].min() - 0.3, Z[:, 0].max() + 0.3, 9)
z2_range = np.linspace(Z[:, 1].min() - 0.3, Z[:, 1].max() + 0.3, 9)

fig, axes = plt.subplots(9, 9, figsize=(12, 12))
fig.suptitle("1a — Item 4: Full Latent Space Grid\n"
             "Each cell decoded from a (z1,z2) point not seen during training",
             fontsize=10)

for i, z2 in enumerate(reversed(z2_range)):
    for j, z1 in enumerate(z1_range):
        decoded = best_ae.decode(np.array([[z1, z2]]))
        decoded_bin = (decoded >= 0.5).astype(int)
        axes[i][j].imshow(decoded_bin.reshape(7, 5), cmap='binary', vmin=0, vmax=1)
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])

plt.tight_layout()
plt.savefig("1a_latent_grid.png", dpi=150)
plt.close()
print("Saved: 1a_latent_grid.png")

print("\n✓ Exercise 1a complete.")
