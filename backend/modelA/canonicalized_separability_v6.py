import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

DATA_PATH = Path(
    "backend/data/modelA/modelA_v6_full.npz"
)

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y = data["y"]
classes = np.asarray(data["classes"])

print()
print("MODEL A V6 CANONICALIZED FEATURE SEPARABILITY AUDIT")
print("=" * 80)

print(f"Total samples: {len(X)}")

# =========================================================
# ONLY USE REAL HAND SAMPLES
# =========================================================

valid = X[:, 93] > 0.5

landmarks = X[valid, :63].reshape(-1, 21, 3)
labels = y[valid]

print(f"Hand-present samples: {len(landmarks)}")

# =========================================================
# CANONICALIZATION
# =========================================================

def canonicalize(hand):

    wrist = hand[0]

    index_mcp = hand[5]
    middle_mcp = hand[9]

    centered = hand - wrist

    # X axis = wrist -> index MCP
    x_axis = index_mcp - wrist

    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-8:
        return None

    x_axis = x_axis / x_norm

    # Y axis = middle MCP direction orthogonalized
    middle_direction = middle_mcp - wrist

    y_axis = (
        middle_direction
        - np.dot(
            middle_direction,
            x_axis
        ) * x_axis
    )

    y_norm = np.linalg.norm(y_axis)

    if y_norm < 1e-8:
        return None

    y_axis = y_axis / y_norm

    # Z axis
    z_axis = np.cross(
        x_axis,
        y_axis
    )

    z_norm = np.linalg.norm(z_axis)

    if z_norm < 1e-8:
        return None

    z_axis = z_axis / z_norm

    # Re-orthogonalize Y
    y_axis = np.cross(
        z_axis,
        x_axis
    )

    y_axis = y_axis / np.linalg.norm(y_axis)

    canonical = np.stack(
        [
            centered @ x_axis,
            centered @ y_axis,
            centered @ z_axis
        ],
        axis=1
    )

    return canonical.astype(np.float32)


canonical = []
keep = []

for i, hand in enumerate(landmarks):

    result = canonicalize(hand)

    if result is not None:
        canonical.append(result)
        keep.append(i)

canonical = np.asarray(
    canonical,
    dtype=np.float32
)

labels = labels[
    np.asarray(keep)
]

print(f"Canonicalized samples: {len(canonical)}")

# =========================================================
# BUILD A PURE GEOMETRIC REPRESENTATION
#
# We deliberately start with normalized/canonicalized XYZ.
# This lets us test the core representation directly.
# =========================================================

canonical_xyz = canonical.reshape(
    len(canonical),
    -1
)

print(
    f"Canonical XYZ shape: "
    f"{canonical_xyz.shape}"
)

# =========================================================
# CLASS CENTROIDS
# =========================================================

print()
print("CLASS CENTROID ANALYSIS")
print("-" * 80)

centroids = {}

for class_id, class_name in enumerate(classes):

    mask = labels == class_id

    if not np.any(mask):
        continue

    centroid = canonical_xyz[mask].mean(
        axis=0
    )

    centroids[class_id] = centroid

print(
    f"Class centroids calculated: "
    f"{len(centroids)}"
)

# =========================================================
# NEAREST CLASS CENTROID
# =========================================================

print()
print("NEAREST CLASS CENTROID ANALYSIS")
print("-" * 80)

centroid_ids = sorted(
    centroids.keys()
)

centroid_matrix = np.asarray(
    [
        centroids[i]
        for i in centroid_ids
    ]
)

nearest_pairs = []

for i, class_id in enumerate(centroid_ids):

    distances = np.linalg.norm(
        centroid_matrix
        - centroid_matrix[i],
        axis=1
    )

    distances[i] = np.inf

    nearest_index = np.argmin(
        distances
    )

    nearest_class_id = centroid_ids[
        nearest_index
    ]

    nearest_distance = distances[
        nearest_index
    ]

    nearest_pairs.append(
        (
            nearest_distance,
            class_id,
            nearest_class_id
        )
    )

nearest_pairs.sort()

for distance, class_id, nearest_class_id in nearest_pairs:

    print(
        f"{str(classes[class_id]):<10} "
        f"-> "
        f"{str(classes[nearest_class_id]):<10} "
        f"distance={distance:.6f}"
    )

# =========================================================
# SPECIFIC M/N ANALYSIS
# =========================================================

print()
print("M / N CENTROID ANALYSIS")
print("-" * 80)

m_id = None
n_id = None

for class_id, class_name in enumerate(classes):

    if str(class_name) == "M":
        m_id = class_id

    if str(class_name) == "N":
        n_id = class_id

if m_id is not None and n_id is not None:

    mn_distance = np.linalg.norm(
        centroids[m_id]
        - centroids[n_id]
    )

    print(
        f"M centroid -> N centroid distance: "
        f"{mn_distance:.6f}"
    )

# =========================================================
# WITHIN-CLASS VS BETWEEN-CLASS DISTANCE
# =========================================================

print()
print("WITHIN-CLASS VS BETWEEN-CLASS DISTANCE")
print("-" * 80)

rng = np.random.default_rng(42)

within_distances = []
between_distances = []

# Sample classes to keep computation manageable.
for class_id in range(len(classes)):

    indices = np.where(
        labels == class_id
    )[0]

    if len(indices) < 2:
        continue

    sample_count = min(
        100,
        len(indices)
    )

    sampled = rng.choice(
        indices,
        size=sample_count,
        replace=False
    )

    points = canonical_xyz[
        sampled
    ]

    # Within-class distances.
    for i in range(
        min(100, len(points))
    ):

        j = rng.integers(
            0,
            len(points)
        )

        if i != j:

            distance = np.linalg.norm(
                points[i]
                - points[j]
            )

            within_distances.append(
                distance
            )

# Between-class distances.
for _ in range(5000):

    class_a, class_b = rng.choice(
        len(classes),
        size=2,
        replace=False
    )

    indices_a = np.where(
        labels == class_a
    )[0]

    indices_b = np.where(
        labels == class_b
    )[0]

    if (
        len(indices_a) == 0
        or len(indices_b) == 0
    ):
        continue

    a = canonical_xyz[
        rng.choice(indices_a)
    ]

    b = canonical_xyz[
        rng.choice(indices_b)
    ]

    between_distances.append(
        np.linalg.norm(a - b)
    )

within_distances = np.asarray(
    within_distances
)

between_distances = np.asarray(
    between_distances
)

print(
    f"Mean within-class distance : "
    f"{within_distances.mean():.6f}"
)

print(
    f"Mean between-class distance: "
    f"{between_distances.mean():.6f}"
)

print(
    f"Median within-class distance : "
    f"{np.median(within_distances):.6f}"
)

print(
    f"Median between-class distance: "
    f"{np.median(between_distances):.6f}"
)

# =========================================================
# SILHOUETTE SCORE
#
# Positive = classes have some separation.
# Near zero = strong overlap.
# Negative = many samples may be closer to another class.
#
# Use a manageable random subset.
# =========================================================

print()
print("SILHOUETTE ANALYSIS")
print("-" * 80)

sample_size = min(
    5000,
    len(canonical_xyz)
)

sample_indices = rng.choice(
    len(canonical_xyz),
    size=sample_size,
    replace=False
)

silhouette = silhouette_score(
    canonical_xyz[sample_indices],
    labels[sample_indices]
)

print(
    f"Samples used: {sample_size}"
)

print(
    f"Silhouette score: "
    f"{silhouette:.6f}"
)

# =========================================================
# NEAREST NEIGHBOR CLASS AGREEMENT
# =========================================================

print()
print("NEAREST-NEIGHBOR CLASS AGREEMENT")
print("-" * 80)

nn_sample_size = min(
    10000,
    len(canonical_xyz)
)

nn_indices = rng.choice(
    len(canonical_xyz),
    size=nn_sample_size,
    replace=False
)

nn = NearestNeighbors(
    n_neighbors=2,
    n_jobs=-1
)

nn.fit(canonical_xyz)

_, neighbor_indices = nn.kneighbors(
    canonical_xyz[nn_indices]
)

nearest = neighbor_indices[:, 1]

nn_accuracy = np.mean(
    labels[nn_indices]
    == labels[nearest]
)

print(
    f"Samples tested: {nn_sample_size}"
)

print(
    f"Nearest-neighbor same-class rate: "
    f"{nn_accuracy * 100:.4f}%"
)

print()
print("AUDIT COMPLETE")
print()
print("IMPORTANT:")
print("No classifier was trained.")
print("V6 was not modified.")
print("This measures whether canonicalized geometry")
print("contains useful class-separating information.")
