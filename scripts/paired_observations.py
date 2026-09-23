"""Canonicalize only sparse, one-level RGB rendering roundoff in paired resets."""

import json

import numpy as np


def align_initial_observation(reference, current):
    with np.load(reference, allow_pickle=False) as source:
        baseline = {k: source[k] for k in source.files}
    with np.load(current, allow_pickle=False) as source:
        observed = {k: source[k] for k in source.files}
    if baseline.keys() != observed.keys():
        raise ValueError("paired initial observation fields differ")
    differences = {}
    for key, expected in baseline.items():
        actual = observed[key]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError("paired initial observation shape/dtype differs")
        if np.array_equal(expected, actual):
            continue
        if not key.endswith("_rgb") or expected.dtype != np.uint8:
            raise ValueError(f"paired initial observations differ: {key}")
        delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        changed = int(np.count_nonzero(delta))
        if delta.max() > 1 or changed / delta.size > 0.0001:
            raise ValueError(f"paired RGB differences exceed render-roundoff limit: {key}")
        differences[key] = {"changed_components": changed, "total_components": int(delta.size),
                            "max_absolute_difference": int(delta.max())}
    if differences:
        # A renderer tolerance must never conceal a different simulator state.
        audits = [json.loads((p.parent / "scene_audit.json").read_text())
                  for p in (reference, current)]
        if audits[0]["case"] != audits[1]["case"] or not np.array_equal(
            audits[0]["physical_state"], audits[1]["physical_state"]
        ):
            raise ValueError("paired physical state/case differs")
        raw = current.with_name(current.stem + "_raw_render.npz")
        if raw.exists():
            raise ValueError("raw render archive already exists")
        current.rename(raw)
        np.savez_compressed(current, **baseline)
        record = {"rule": "RGB <=1/255 in <=0.01% components per view; all other arrays and physical state exact",
                  "differences": differences, "raw_render": raw.name,
                  "canonical_source": str(reference)}
        (current.parent / "initial_render_alignment.json").write_text(json.dumps(record, indent=2)+"\n")
    return differences
