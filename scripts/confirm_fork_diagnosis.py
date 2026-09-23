"""One locked confirmation of predeclared mechanisms, without model selection."""

import argparse
import json

import torch
from fork_predictor_lab import OUT, dataset, evaluate, make_model, teacher_probe
from gpu_devices import verify_cuda_target
from run_real_predictor_debug import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    a = p.parse_args()
    verify_cuda_target(a.gpu)
    torch.set_num_threads(2)
    plan = json.loads((OUT / "confirmation_protocol.json").read_text())
    cases = dataset(confirmation=True)
    if [c["case"]["case_id"] for c in cases] != plan["confirmation_cases"]:
        raise ValueError("confirmation identities changed")
    results = []
    for identifier in plan["models"][a.shard :: 2]:
        if identifier == "original stage2 best":
            model = make_model("original", 17)
            job = {"variant": "original", "blind": False, "source": "stage2 best"}
        else:
            payload = torch.load(
                OUT / identifier, map_location="cpu", weights_only=True
            )
            job = payload["job"]
            model = make_model(job["variant"], job["manual_seed"])
            model.load_state_dict(payload["state"], strict=True)
        result = {
            "model": identifier,
            "job": job,
            "confirmation": evaluate(model, cases, job["blind"]),
        }
        results.append(result)
        save(
            OUT / f"confirmation_{a.shard}.json",
            {"complete": False, "results": results},
        )
        del model
        torch.cuda.empty_cache()
    save(OUT / f"confirmation_{a.shard}.json", {"complete": True, "results": results})
    if a.shard == 0:
        train = [c for c in dataset() if c["case"]["split"] == "train"]
        probe = teacher_probe(train + cases)
        # The probe's second split contains confirmation only in this call.
        for values in probe["branch_accuracy"].values():
            values["confirmation"] = values.pop("development")
        save(OUT / "teacher_confirmation_probe.json", probe)


if __name__ == "__main__":
    main()
