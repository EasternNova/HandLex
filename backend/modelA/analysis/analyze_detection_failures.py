
import json
from pathlib import Path


# Resolve the Model A directory independently of the terminal location
MODEL_A_DIR = Path(__file__).resolve().parents[1]

METADATA_PATH = MODEL_A_DIR / "data" / "modelA_v6_full.json"

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"V6 metadata not found: {METADATA_PATH}")

with open(METADATA_PATH, "r", encoding="utf-8-sig") as file:
    data = json.load(file)

print("=" * 70)
print("MODEL A V6 DETECTION FAILURE AUDIT")
print("=" * 70)

print(f"Total images       : {data['total_images']}")
print(f"Successful samples : {data['successful_samples']}")
print(f"Failures           : {data['failures']}")
print(f"Detection rate     : {data['detection_rate']:.4%}")

print("\nPER-CLASS DETECTION")
print("-" * 70)

results = []

for cls, stats in data["class_stats"].items():
    success = stats["success"]
    failure = stats["failure"]
    total = success + failure
    rate = success / total if total else 0.0

    print(
        f"{cls:8s} "
        f"success={success:4d} "
        f"failure={failure:4d} "
        f"rate={rate:7.2%}"
    )

    results.append((rate, cls, success, failure))

print("\nLOWEST DETECTION CLASSES")
print("-" * 70)

for rate, cls, success, failure in sorted(results)[:10]:
    print(
        f"{cls:8s} "
        f"{rate:7.2%} "
        f"success={success:4d} "
        f"failure={failure:4d}"
    )

print("\nSPACE AUDIT")
print("-" * 70)

space = data["class_stats"].get("space")

if space is not None:
    total = space["success"] + space["failure"]
    rate = space["success"] / total if total else 0.0

    print(f"Space successful : {space['success']}")
    print(f"Space failures   : {space['failure']}")
    print(f"Space detection  : {rate:.2%}")
else:
    print("No 'space' class found in metadata.")

print("\nAUDIT COMPLETE")
