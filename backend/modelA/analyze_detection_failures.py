from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "modelA"
    / "modelA_v6_full.json"
)

with open(METADATA, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("MODEL A V6 DETECTION FAILURE AUDIT")
print("=" * 70)

print(f"Total images       : {data['total_images']}")
print(f"Successful samples : {data['successful_samples']}")
print(f"Failures           : {data['failures']}")
print(f"Detection rate     : {data['detection_rate']:.4%}")

print("\nPER-CLASS DETECTION")
print("-" * 70)

for cls, stats in data["class_stats"].items():
    total = stats["success"] + stats["failure"]
    rate = stats["success"] / total if total else 0

    print(
        f"{cls:8s} "
        f"success={stats['success']:4d} "
        f"failure={stats['failure']:4d} "
        f"rate={rate:7.2%}"
    )

print("\nLOWEST DETECTION CLASSES")
print("-" * 70)

results = []

for cls, stats in data["class_stats"].items():
    total = stats["success"] + stats["failure"]
    rate = stats["success"] / total if total else 0
    results.append((rate, cls, stats["success"], stats["failure"]))

for rate, cls, success, failure in sorted(results)[:10]:
    print(
        f"{cls:8s} "
        f"{rate:7.2%} "
        f"success={success:4d} "
        f"failure={failure:4d}"
    )

print("\nSPACE AUDIT")
print("-" * 70)

space = data["class_stats"]["space"]

print(f"Space successful : {space['success']}")
print(f"Space failures   : {space['failure']}")
print(
    f"Space detection  : "
    f"{space['success'] / (space['success'] + space['failure']):.2%}"
)

print("\nAUDIT COMPLETE")