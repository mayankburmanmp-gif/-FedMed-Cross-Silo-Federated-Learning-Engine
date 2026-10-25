
"""Compare centralized and federated Dice scores."""

import json
import config as cfg

baseline_file = cfg.OUT_DIR / "baseline_metrics.json"
federated_file = cfg.OUT_DIR / "federated_metrics.json"

# Check that both files exist
if not baseline_file.exists():
    print(f"Missing file: {baseline_file}")
    raise SystemExit(1)

if not federated_file.exists():
    print(f"Missing file: {federated_file}")
    raise SystemExit(1)

# Load JSON data
base = json.loads(baseline_file.read_text())
fed = json.loads(federated_file.read_text())

# Extract centralized score
baseline = base.get("best", {})
baseline_dice = baseline.get("dice_mean")

if baseline_dice is None:
    print("Error: dice_mean is missing from baseline metrics.")
    print("Check baseline_metrics.json.")
    raise SystemExit(1)

# Extract federated scores
fed_scores = []

if isinstance(fed, list):
    for item in fed:
        if isinstance(item, dict):
            score = item.get("val_dice_mean")
            if score is not None:
                fed_scores.append(float(score))

if not fed_scores:
    print("No val_dice_mean scores found in federated_metrics.json.")
    print("Check the JSON structure and run federated training first.")
    print("Expected key: val_dice_mean")
    raise SystemExit(1)

# Compare results
federated_dice = fed_scores[-1]
gap = baseline_dice - federated_dice

print(f"Centralized best mean Dice : {baseline_dice:.4f}")
print(f"Federated final mean Dice  : {federated_dice:.4f}")
print(f"Gap                       : {gap:+.4f}")
print("Per-round federated Dice  :", ", ".join(
    f"{score:.3f}" for score in fed_scores
))
