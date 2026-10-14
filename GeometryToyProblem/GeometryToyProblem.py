
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    generic_filter,
    label,
    binary_erosion
)


# ============================================================
# PARAMETERS
# ============================================================

WIDTH = 600
HEIGHT = 400

# Background radiometry
BACKGROUND_MEAN = 1720
BACKGROUND_STD = 0

# Object radiometry
OBJECT_DN = 2000
OBJECT_STD = 0

# Detector noise
GAUSSIAN_NOISE_STD = 3

# Local contrast threshold
K_THRESHOLD = 3.5

# Number of angular measurements
N_ANGLES = 720

# Radial sampling resolution
DR = 0.25

# Minimum number of pixels required for a detected object
MIN_OBJECT_PIXELS = 100

np.random.seed(42)


# ============================================================
# CREATE IMAGE COORDINATES
# ============================================================

y, x = np.mgrid[0:HEIGHT, 0:WIDTH]


# ============================================================
# CREATE BACKGROUND
# ============================================================

gradient = (
    120 * (x / WIDTH) +
    60 * (y / HEIGHT)
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


# ============================================================
# SHAPE CREATION FUNCTIONS
# ============================================================

def create_circle(cx, cy, radius):

    return (
        (x - cx)**2 +
        (y - cy)**2
    ) <= radius**2


def create_rectangle(
    cx,
    cy,
    width,
    height
):

    return (
        (np.abs(x - cx) <= width / 2) &
        (np.abs(y - cy) <= height / 2)
    )


def create_regular_polygon(
    cx,
    cy,
    radius,
    n_sides,
    rotation=np.pi / 2
):

    dx = x - cx
    dy = y - cy

    theta = np.arctan2(
        dy,
        dx
    )

    sector_angle = (
        2 * np.pi / n_sides
    )

    relative_angle = (
        theta -
        rotation +
        sector_angle / 2
    )

    relative_angle = (
        relative_angle +
        np.pi
    ) % (
        2 * np.pi
    ) - np.pi

    relative_angle = (
        relative_angle +
        sector_angle / 2
    ) % sector_angle - (
        sector_angle / 2
    )

    polygon_radius = (
        radius *
        np.cos(sector_angle / 2)
        /
        np.cos(relative_angle)
    )

    pixel_radius = np.sqrt(
        dx**2 +
        dy**2
    )

    return (
        pixel_radius <=
        polygon_radius
    )


# ============================================================
# CREATE TRUE OBJECTS
# ============================================================

true_shapes = []


# Circle
true_shapes.append({
    "name": "Circle",
    "mask": create_circle(
        100,
        100,
        45
    )
})


# Square
true_shapes.append({
    "name": "Square",
    "mask": create_rectangle(
        300,
        100,
        85,
        85
    )
})


# Rectangle
true_shapes.append({
    "name": "Rectangle",
    "mask": create_rectangle(
        500,
        100,
        110,
        65
    )
})


# Triangle
true_shapes.append({
    "name": "Triangle",
    "mask": create_regular_polygon(
        170,
        290,
        60,
        3
    )
})


# Pentagon
true_shapes.append({
    "name": "Pentagon",
    "mask": create_regular_polygon(
        410,
        285,
        60,
        5
    )
})


# ============================================================
# INSERT OBJECTS INTO RADIOMETRIC SCENE
# ============================================================

true_mask = np.zeros(
    (HEIGHT, WIDTH),
    dtype=bool
)

for shape in true_shapes:

    mask = shape["mask"]

    object_noise = np.random.normal(
        0,
        OBJECT_STD,
        (HEIGHT, WIDTH)
    )

    scene[mask] = (
        OBJECT_DN +
        object_noise[mask]
    )

    true_mask |= mask


# ============================================================
# ADD DETECTOR NOISE
# ============================================================

detector_noise = np.random.normal(
    0,
    GAUSSIAN_NOISE_STD,
    (HEIGHT, WIDTH)
)

noisy_scene = (
    scene +
    detector_noise
)


# ============================================================
# LOCAL BACKGROUND ESTIMATION
# ============================================================
#
# For every pixel, calculate the mean of the 8 surrounding
# pixels while excluding the center pixel itself.
#
#                 N
#              NW   NE
#            W   X   E
#              SW SE
#                 S
#
# X = pixel being tested
#
# DN_bkg = average of the surrounding pixels
# ============================================================

def neighbor_mean(values):

    center = len(values) // 2

    neighbors = np.delete(
        values,
        center
    )

    return np.mean(
        neighbors
    )


DN_bkg = generic_filter(
    noisy_scene,
    neighbor_mean,
    size=3,
    mode="reflect"
)


# ============================================================
# LOCAL CONTRAST
# ============================================================
#
# Positive contrast means the pixel is brighter than its
# immediate neighborhood.
#
#               DN_contrast
#                    =
#
#             DN_pixel - DN_local_background
# ============================================================

DN_contrast = (
    noisy_scene -
    DN_bkg
)


# ============================================================
# LOCAL-CONTRAST DETECTOR
# ============================================================
#
# Detection threshold:
#
#             T = k * sigma_noise
#
# A pixel is declared an object pixel when:
#
#             DN_contrast > T
#
# This is now the ACTUAL OBJECT DETECTOR.
# ============================================================

sigma_n = GAUSSIAN_NOISE_STD

contrast_threshold = (
    K_THRESHOLD *
    sigma_n
)

detected_mask = (
    DN_contrast >
    contrast_threshold
)


# ============================================================
# CONNECTED COMPONENT DETECTION
# ============================================================

labeled_objects, number_objects = label(
    detected_mask
)


# ============================================================
# REMOVE SMALL DETECTIONS
# ============================================================

clean_detected_mask = np.zeros_like(
    detected_mask
)

for object_id in range(
    1,
    number_objects + 1
):

    component = (
        labeled_objects ==
        object_id
    )

    pixel_count = np.sum(
        component
    )

    if pixel_count >= MIN_OBJECT_PIXELS:

        clean_detected_mask |= component


# Re-label after removing small detections

labeled_objects, number_objects = label(
    clean_detected_mask
)


# ============================================================
# RADIAL SIGNATURE
# ============================================================

def calculate_radial_signature(
    object_mask
):

    # --------------------------------------------------------
    # Find object pixels
    # --------------------------------------------------------

    object_y, object_x = np.nonzero(
        object_mask
    )

    if len(object_x) == 0:

        return (
            np.array([]),
            np.array([]),
            np.nan,
            np.nan
        )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    centroid_x = np.mean(
        object_x
    )

    centroid_y = np.mean(
        object_y
    )

    # --------------------------------------------------------
    # Angular sampling
    # --------------------------------------------------------

    angles = np.linspace(
        0,
        2 * np.pi,
        N_ANGLES,
        endpoint=False
    )

    # --------------------------------------------------------
    # Maximum possible radius
    # --------------------------------------------------------

    max_radius = np.sqrt(
        max(
            centroid_x,
            WIDTH - centroid_x
        )**2
        +
        max(
            centroid_y,
            HEIGHT - centroid_y
        )**2
    )

    r_samples = np.arange(
        0,
        max_radius,
        DR
    )

    radial_distance = np.full(
        N_ANGLES,
        np.nan
    )

    # --------------------------------------------------------
    # Cast rays
    # --------------------------------------------------------

    for i, theta in enumerate(angles):

        #Calculates x-coordinate of points along a ray,
        #starting from the centroid and extending outward.
        ray_x = (
            centroid_x +
            r_samples *
            np.cos(theta)
        )
        #Same thing but for y-coordinates.
        ray_y = (
            centroid_y +
            r_samples *
            np.sin(theta)
        )

        #Convert floats to integers

        ix = np.rint(
            ray_x
        ).astype(int)

        iy = np.rint(
            ray_y
        ).astype(int)

        #Indices of which integer coordinates are within image bounds.
        valid = (
            (ix >= 0) &
            (ix < WIDTH) &
            (iy >= 0) &
            (iy < HEIGHT)
        )

        #Place only valid rays
        ix = ix[valid]
        iy = iy[valid]

        #Will only consider radii corresponding to valid rays.
        radii = r_samples[:len(ix)]

        #Get the object mask at valid coordinates.
        inside = object_mask[
            iy,
            ix
        ]

        #Get indices of coordinates inside object mask
        inside_indices = np.where(inside)[0]

        #If indices within the object mask,
        #Get last index and its corresponding radius
        if len(inside_indices) > 0:

            last_inside = (
                inside_indices[-1]
            )

            radial_distance[i] = (
                radii[last_inside]
            )

    return (
        angles,
        radial_distance,
        centroid_x,
        centroid_y
    )


# ============================================================
# SHAPE CLASSIFICATION
# ============================================================

def classify_signature(
    angles,
    radial_distance
):

    #Basic validation checks.
    #If fewer than 10 radial distances are valid, return Unknown.
    valid = np.isfinite(radial_distance)

    r = radial_distance[valid]

    if len(r) < 10:
        return "Unknown"

    # --------------------------------------------------------
    # Scale-normalized radial signal
    # --------------------------------------------------------

    mean_r = np.mean(r)

    normalized_r = (
        r / mean_r
    )

    # --------------------------------------------------------
    # Circle test
    # --------------------------------------------------------

    variation = np.std(
        normalized_r
    )

    if variation < 0.025:

        return "Circle"

    # --------------------------------------------------------
    # Fourier analysis
    # --------------------------------------------------------

    centered_r = (
        normalized_r -
        np.mean(normalized_r)
    )

    spectrum = np.abs(
        np.fft.rfft(
            centered_r
        )
    )

    spectrum[0] = 0

    dominant_frequency = (
        np.argmax(spectrum)
    )

    shape_map = {
        3: "Triangle",
        4: "Square",
        5: "Pentagon",
        6: "Hexagon",
        7: "7-sided",
        8: "8-sided"
    }

    return shape_map.get(
        dominant_frequency,
        f"{dominant_frequency}-fold"
    )


# ============================================================
# ANALYZE DETECTED OBJECTS
# ============================================================

results = []


for object_id in range(
    1,
    number_objects + 1
):

    object_mask = (
        labeled_objects ==
        object_id
    )

    pixel_count = np.sum(
        object_mask
    )

    if pixel_count < MIN_OBJECT_PIXELS:

        continue

    (
        angles,
        radial_distance,
        centroid_x,
        centroid_y
    ) = calculate_radial_signature(
        object_mask
    )

    classification = (
        classify_signature(
            angles,
            radial_distance
        )
    )

    valid = np.isfinite(
        radial_distance
    )

    mean_r = np.mean(
        radial_distance[valid]
    )

    std_r = np.std(
        radial_distance[valid]
    )

    normalized_variation = (
        std_r /
        mean_r
    )

    results.append({
        "id": object_id,
        "mask": object_mask,
        "angles": angles,
        "r": radial_distance,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "classification": classification,
        "mean_r": mean_r,
        "variation":
            normalized_variation
    })


# ============================================================
# DETECTION PERFORMANCE
# ============================================================
#
# Compare the local-contrast detector against the known
# synthetic truth.
# ============================================================

true_positive_pixels = np.sum(
    clean_detected_mask &
    true_mask
)

false_positive_pixels = np.sum(
    clean_detected_mask &
    ~true_mask
)

missed_pixels = np.sum(
    true_mask &
    ~clean_detected_mask
)

Pd = (
    true_positive_pixels /
    np.sum(true_mask)
)

Pfa = (
    false_positive_pixels /
    np.sum(~true_mask)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("LOCAL-CONTRAST OBJECT DETECTOR")
print("========================================")

print(
    f"Contrast threshold = "
    f"{contrast_threshold:.2f} DN"
)

print(
    f"Detected objects   = "
    f"{len(results)}"
)

print(
    f"Pixel Pd           = "
    f"{Pd:.4f}"
)

print(
    f"Pixel Pfa          = "
    f"{Pfa:.6f}"
)


print("\nDetected Shape Results")
print("======================")

print(
    f"{'ID':>4} "
    f"{'Centroid X':>12} "
    f"{'Centroid Y':>12} "
    f"{'Mean r':>10} "
    f"{'Variation':>12} "
    f"{'Detected Shape':>18}"
)

print("-" * 80)

for result in results:

    print(
        f"{result['id']:4d} "
        f"{result['centroid_x']:12.2f} "
        f"{result['centroid_y']:12.2f} "
        f"{result['mean_r']:10.2f} "
        f"{result['variation']:12.4f} "
        f"{result['classification']:>18}"
    )


# ============================================================
# FIGURE 1:
# RADIOMETRIC IMAGE
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    noisy_scene,
    cmap="gray"
)

plt.title(
    "Noisy Radiometric Scene"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 2:
# LOCAL CONTRAST
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    DN_contrast,
    cmap="gray"
)

plt.colorbar(
    label="Local contrast [DN]"
)

plt.title(
    "Local Contrast Image"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 3:
# LOCAL-CONTRAST DETECTION MASK
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    clean_detected_mask,
    cmap="gray"
)

plt.title(
    f"Local-Contrast Detection "
    f"(k = {K_THRESHOLD})"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 4:
# DETECTED OBJECTS + CENTROIDS
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.imshow(
    noisy_scene,
    cmap="gray"
)

for result in results:

    cx = result["centroid_x"]
    cy = result["centroid_y"]

    plt.contour(
        result["mask"],
        levels=[0.5],
        linewidths=2
    )

    plt.scatter(
        cx,
        cy,
        s=100,
        marker="+"
    )

    plt.text(
        cx + 8,
        cy - 8,
        (
            f"ID {result['id']}\n"
            f"{result['classification']}"
        ),
        fontsize=10,
        bbox=dict(
            facecolor="white",
            alpha=0.7
        )
    )

plt.title(
    "Objects Detected from Local Contrast"
)

plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()


# ============================================================
# FIGURE 5:
# RAW RADIAL SIGNATURES
# ============================================================

plt.figure(
    figsize=(12, 7)
)

for result in results:

    angle_degrees = np.degrees(
        result["angles"]
    )

    plt.plot(
        angle_degrees,
        result["r"],
        linewidth=2,
        label=(
            f"ID {result['id']} — "
            f"{result['classification']}"
        )
    )

plt.xlabel(
    "Angle θ (degrees)"
)

plt.ylabel(
    "Boundary distance r(θ) [pixels]"
)

plt.title(
    "Detected Objects — Radial Signatures"
)

plt.xlim(
    0,
    360
)

plt.grid(True)

plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# FIGURE 7:
# POLAR SIGNATURE OVERLAY
# ============================================================

fig = plt.figure(
    figsize=(9, 9)
)

ax = fig.add_subplot(
    111,
    projection="polar"
)

for result in results:

    valid = np.isfinite(
        result["r"]
    )

    mean_r = np.mean(
        result["r"][valid]
    )

    normalized_r = (
        result["r"] /
        mean_r
    )

    ax.plot(
        result["angles"],
        normalized_r,
        linewidth=2,
        label=(
            f"ID {result['id']} — "
            f"{result['classification']}"
        )
    )

ax.set_title(
    "Normalized Radial Shape Signatures",
    pad=25
)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.10)
)

plt.tight_layout()
plt.show()