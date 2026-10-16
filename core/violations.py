"""Configurable PPE violation rules for positive- and negative-label datasets."""

from __future__ import annotations

import re
from typing import Any

NEGATIVE_TERMS = ("no-", "no_", "without", "missing", "unprotected", "absence")
PPE_TERMS = ("helmet", "hardhat", "hard_hat", "vest", "mask", "goggle", "glove", "boot", "shoe", "harness")


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def is_negative_ppe_label(label: str) -> bool:
    value = label.lower().replace(" ", "_")
    return any(term in value for term in NEGATIVE_TERMS) and any(term in value for term in PPE_TERMS)


def negative_label_target(label: str) -> str | None:
    normalized = normalize_label(label)
    for prefix in ("without", "unprotected", "absence", "missing", "no"):
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return normalized[len(prefix):]
    return None


def default_required_classes(class_names: list[str]) -> list[str]:
    positives: list[str] = []
    for name in class_names:
        normalized = normalize_label(name)
        if normalized == "person" or is_negative_ppe_label(name):
            continue
        if any(term in normalized for term in PPE_TERMS):
            paired_negative = any(
                normalize_label(negative_name).replace("no", "") in normalized
                or normalized in normalize_label(negative_name).replace("no", "")
                for negative_name in class_names
                if is_negative_ppe_label(negative_name)
            )
            if not paired_negative:
                positives.append(name)
    return positives


def evaluate_violations(
    detections: list[dict[str, Any]],
    required_classes: list[str],
    negative_mode: bool | None = None,
    class_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate negative labels directly and check unmatched positive classes by box-center inclusion."""
    if negative_mode is None:
        negative_mode = any(is_negative_ppe_label(item["class_name"]) for item in detections)
    violations: list[dict[str, Any]] = []
    if negative_mode:
        for detection in detections:
            if is_negative_ppe_label(detection["class_name"]):
                violations.append({**detection, "violation_type": detection["class_name"]})
    declared_classes = class_names or [item["class_name"] for item in detections]
    negative_targets = {
        target
        for name in declared_classes
        if is_negative_ppe_label(name)
        if (target := negative_label_target(name)) is not None
    }
    positive_requirements = [
        required
        for required in required_classes
        if normalize_label(required) not in negative_targets
    ]

    people = [item for item in detections if normalize_label(item["class_name"]) == "person"]
    ppe_detections = [item for item in detections if item["class_name"] in positive_requirements]
    for person in people:
        px1, py1, px2, py2 = person["bbox"]
        for required in positive_requirements:
            normalized_required = normalize_label(required)
            present = False
            for ppe in ppe_detections:
                if ppe["class_name"] != required:
                    continue
                x1, y1, x2, y2 = ppe["bbox"]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                    present = True
                    break
            if not present:
                label = f"Missing {required}"
                violations.append(
                    {
                        "class_name": label,
                        "violation_type": label,
                        "confidence": float(person["confidence"]),
                        "bbox": person["bbox"],
                        "source_class": normalized_required,
                    }
                )
    return violations
