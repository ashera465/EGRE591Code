import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.ndimage import binary_erosion

# ============================================================
# PARAMETERS
# ============================================================

WIDTH = 400
HEIGHT = 300

BACKGROUND_MEAN = 1720
BACKGROUND_STD = 0

OBJECT_DN = 1800
OBJECT_STD = 0

CIRCLE_CENTER = (280, 150)
CIRCLE_RADIUS = 45

GAUSSIAN_NOISE_STD = 3

np.random.seed(42)

# ============================================================
# CREATE SYNTHETIC SCENE
# ============================================================

y, x = np.mgrid[0:HEIGHT, 0:WIDTH]

gradient = (
    40 * (x / WIDTH) +
    30 * (y / HEIGHT)
)

background_noise = np.random.normal(
    0,
    BACKGROUND_STD,
    (HEIGHT, WIDTH)
)

scene = (
    BACKGROUND_MEAN +
    gradient +
    background_noise
)

# Circular target
cx, cy = CIRCLE_CENTER

target_mask = (
    (x - cx)**2 +
    (y - cy)**2
) <= CIRCLE_RADIUS**2

object_noise = np.random.normal(
    0,
    OBJECT_STD,
    (HEIGHT, WIDTH)
)

scene[target_mask] = (
    OBJECT_DN +
    object_noise[target_mask]
)

# ============================================================
# ADD GAUSSIAN NOISE
# ============================================================

detector_noise = np.random.normal(
    0,
    GAUSSIAN_NOISE_STD,
    (HEIGHT, WIDTH)
)

noisy_scene = scene + detector_noise

# ============================================================
# LOCAL BACKGROUND FROM 8 NEIGHBORS
# ============================================================

def neighbor_mean(values):

    center = len(values) // 2

    neighbors = np.delete(
        values,
        center
    )

    return np.mean(neighbors)


DN_bkg = generic_filter(
    noisy_scene,
    neighbor_mean,
    size=3,
    mode="reflect"
)

# ============================================================
# LOCAL CONTRAST
# ============================================================

DN_contrast = (
    noisy_scene -
    DN_bkg
)

# Known noise standard deviation
sigma_n = GAUSSIAN_NOISE_STD

# ============================================================
# K SWEEP: 1 THROUGH 10
# ============================================================

k_values = np.arange(0, 3,0.2)

Pd_values = []
Pfa_values = []

total_targets = np.sum(target_mask)
total_background = np.sum(~target_mask)

for k in k_values:

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = k * sigma_n

    # --------------------------------------------------------
    # Detection decision
    # --------------------------------------------------------

    detection = (
        DN_contrast >
        threshold
    )

    # --------------------------------------------------------
    # True positives
    # --------------------------------------------------------

    true_positives = np.sum(
        detection & target_mask
    )

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    false_positives = np.sum(
        detection & ~target_mask
    )

    # Target boundary
    target_boundary = (
        target_mask &
        ~binary_erosion(
            target_mask,
            structure=np.ones((3, 3))
        )
    )

    # Probability of detection
    Pd = (
        np.sum(detection & target_boundary)
        /
        np.sum(target_boundary)
    )

    # Probability of false alarm
    Pfa = (
        np.sum(detection & ~target_mask)
        /
        np.sum(~target_mask)
    )

    Pd_values.append(Pd)
    Pfa_values.append(Pfa)

Pd_values = np.array(Pd_values)
Pfa_values = np.array(Pfa_values)
#
# ============================================================
# PRINT RESULTS
# ============================================================

print("\nDetector Performance")
print("====================")

print(
    f"{'k':>4} "
    f"{'Threshold':>12} "
    f"{'Pd':>12} "
    f"{'Pfa':>12}"
)

print("-" * 45)

for k, pd, pfa in zip(
    k_values,
    Pd_values,
    Pfa_values
):

    print(
        f"{k:4f} "
        f"{k * sigma_n:12.2f} "
        f"{pd:12.4f} "
        f"{pfa:12.6f}"
    )

# ============================================================
# FIGURE 1:
# ORIGINAL SCENE AND CONTRAST
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

# ------------------------------------------------------------
# Gaussian noisy scene
# ------------------------------------------------------------

im1 = axes[0].imshow(
    noisy_scene,
    cmap="inferno"
)

axes[0].set_title(
    "Gaussian-Noisy Scene"
)

axes[0].set_xlabel("Pixel X")
axes[0].set_ylabel("Pixel Y")

fig.colorbar(
    im1,
    ax=axes[0],
    label="DN"
)

# ------------------------------------------------------------
# Contrast map
# ------------------------------------------------------------

im2 = axes[1].imshow(
    DN_contrast,
    cmap="RdBu_r"
)

axes[1].set_title(
    "Local Contrast Map"
)

axes[1].set_xlabel("Pixel X")
axes[1].set_ylabel("Pixel Y")

fig.colorbar(
    im2,
    ax=axes[1],
    label="DN Contrast"
)

plt.tight_layout()
plt.show()

# ============================================================
# FIGURE 2:
# ROC CURVE FOR k = 1 TO 10
# ============================================================

plt.figure(figsize=(8, 7))

# Plot ROC points
plt.plot(
    Pfa_values,
    Pd_values,
    marker="o",
    linewidth=2,
    label="Detector"
)

# Label every point with its k value
for k, pfa, pd in zip(
    k_values,
    Pfa_values,
    Pd_values
):

    plt.annotate(
        f"k={k}",
        (pfa, pd),
        xytext=(6, 6),
        textcoords="offset points"
    )

# Chance line
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    linewidth=1.5,
    label="Chance"
)

plt.xlabel(
    r"$P_{fa}$"
)

plt.ylabel(
    r"$P_d$"
)

plt.title(
    "ROC Curve"
)

plt.xlim(0, 1)
plt.ylim(0, 1.05)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# FIGURE 3:
# Pd AND Pfa VS k
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    k_values,
    Pd_values,
    marker="o",
    linewidth=2,
    label=r"$P_d$"
)

plt.plot(
    k_values,
    Pfa_values,
    marker="o",
    linewidth=2,
    label=r"$P_{fa}$"
)

plt.xlabel(
    r"Threshold multiplier $k$"
)

plt.ylabel(
    "Probability"
)

plt.title(
    r"$P_d$ and $P_{fa}$ vs. $k$"
)

plt.xticks(k_values)

plt.ylim(0, 1.05)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# FIGURE 4:
# DETECTION MAP FOR EACH k
# ============================================================

fig, axes = plt.subplots(
    3,
    5,
    figsize=(18, 7)
)

for ax, k, pd, pfa in zip(
    axes.flat,
    k_values,
    Pd_values,
    Pfa_values
):

    threshold = k * sigma_n

    detection = (
        DN_contrast >
        threshold
    )

    # Normalize scene for display
    normalized = (
        noisy_scene -
        noisy_scene.min()
    ) / (
        noisy_scene.max() -
        noisy_scene.min()
    )

    detection_image = np.stack(
        [normalized, normalized, normalized],
        axis=-1
    )

    # Green detected pixels
    detection_image[detection] = [0, 1, 0]

    ax.imshow(
        detection_image
    )

    ax.set_title(
        f"k={k}\n"
        f"Pd={pd:.3f}, Pfa={pfa:.4f}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.suptitle(
    "Detection Maps for k",
    fontsize=16
)

plt.tight_layout()
plt.show()