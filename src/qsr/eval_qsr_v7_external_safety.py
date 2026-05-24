from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

from peft import PeftModel

from qsr.eval_qsr_v6_xstest import (
    SAFE_RECOVERY_TEXT,
    build_prompt,
    classify_refusal,
    generate_completion,
)
from qsr.eval_qsr_v7_xstest import (
    hybrid_route,
    is_low_risk_factual_context,
    is_utility_safe_context,
    should_recover_safe_feature_refusal,
)
from qsr.features_v6 import extract_feature_flags
from qsr.modeling import load_model_and_tokenizer
from qsr.router import ROUTER_REFUSAL_TEXT
from qsr.router_v7 import load_v7_bundle, make_calibrator_feature_rows, score_with_base_router
from qsr.utility_calibrator import load_utility_bundle


@dataclass
class QSRV7ExternalConfig:
    model_name: str
    precision: str
    dataset_kind: str
    dataset_path: str
    text_column: str
    router_v6_path: str
    router_v7_path: str
    safety_adapter_path: str
    factual_adapter_path: str
    threshold: float
    max_samples: int
    max_new_tokens: int
    output_path: str
    trust_remote_code: bool = True
    use_chat_template: bool = True
    use_benign_override: bool = True
    use_risk_override: bool = True
    use_fixed_high_risk_refusal: bool = True
    use_safe_feature_recovery: bool = False
    use_utility_safe_override: bool = False
    use_harmful_intent_blocker: bool = True
    utility_calibrator_path: str = ""
    utility_threshold: float = 0.7
    use_pattern_utility_gate: bool = True
    benign_gate_min: int = 1
    use_low_risk_factual_recovery: bool = False
    factual_recovery_max_risk: float = 0.5
    factual_recovery_max_route_probability: float = 0.5
    safe_recovery_mode: str = "legacy"
    use_semantic_benign_gate: bool = False
    semantic_benign_gate_profile: str = "balanced"
    semantic_benign_gate_max_risk: float = 0.85
    semantic_benign_gate_max_route_probability: float = 0.85


def load_config(path: str | Path) -> QSRV7ExternalConfig:
    with open(path, "r", encoding="utf-8") as f:
        return QSRV7ExternalConfig(**json.load(f))


def load_rows(cfg: QSRV7ExternalConfig) -> list[dict]:
    path = Path(cfg.dataset_path)
    if cfg.dataset_kind == "csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        out = []
        for idx, row in enumerate(rows):
            prompt = row[cfg.text_column]
            if prompt.strip():
                out.append({"id": row.get("BehaviorID") or str(idx + 1), "prompt": prompt, "metadata": row})
        return out[: cfg.max_samples] if cfg.max_samples > 0 else out
    if cfg.dataset_kind == "json_list":
        data = json.load(open(path, "r", encoding="utf-8"))
        out = [{"id": str(idx + 1), "prompt": str(prompt), "metadata": {}} for idx, prompt in enumerate(data) if str(prompt).strip()]
        return out[: cfg.max_samples] if cfg.max_samples > 0 else out
    if cfg.dataset_kind == "json_records":
        data = json.load(open(path, "r", encoding="utf-8"))
        out = []
        for idx, row in enumerate(data):
            prompt = str(row[cfg.text_column])
            if prompt.strip():
                out.append({"id": str(row.get("id") or idx + 1), "prompt": prompt, "metadata": row})
        return out[: cfg.max_samples] if cfg.max_samples > 0 else out
    raise ValueError(f"Unsupported dataset_kind: {cfg.dataset_kind}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.use_semantic_benign_gate:
        raise ValueError(
            "Semantic benign gate is a diagnostic prompt-pattern recovery path and "
            "is not included in the public clean Q-SR release."
        )
    v7_bundle = load_v7_bundle(cfg.router_v7_path)
    utility_bundle = load_utility_bundle(cfg.utility_calibrator_path) if cfg.utility_calibrator_path else None
    rows = load_rows(cfg)
    prompts = [row["prompt"] for row in rows]
    risk_scores, ambiguity_scores, _ = score_with_base_router(cfg.router_v6_path, prompts)
    feature_rows = make_calibrator_feature_rows(prompts, risk_scores, ambiguity_scores)
    route_probs = v7_bundle["calibrator"].predict_proba(feature_rows)[:, 1].tolist()

    base_model, tokenizer = load_model_and_tokenizer(
        cfg.model_name,
        cfg.precision,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, cfg.safety_adapter_path, adapter_name="safety")
    model.load_adapter(cfg.factual_adapter_path, adapter_name="factual")
    model.eval()

    results = []
    started = time.time()
    for row, risk_score, ambiguity_score, route_probability in zip(rows, risk_scores, ambiguity_scores, route_probs):
        prompt = row["prompt"]
        flags = extract_feature_flags(prompt)
        adapter, route_reason = hybrid_route(
            route_probability=route_probability,
            flags=flags,
            threshold=cfg.threshold,
            use_benign_override=cfg.use_benign_override,
            use_risk_override=cfg.use_risk_override,
            use_harmful_intent_blocker=cfg.use_harmful_intent_blocker,
        )
        utility_probability = None
        utility_gate_reason = "disabled"
        if cfg.use_utility_safe_override and adapter == "safety" and route_reason == "calibrator":
            is_utility_safe, utility_probability, utility_gate_reason = is_utility_safe_context(
                prompt=prompt,
                flags=flags,
                risk_score=risk_score,
                ambiguity_score=ambiguity_score,
                utility_bundle=utility_bundle,
                utility_threshold=cfg.utility_threshold,
                use_pattern_utility_gate=cfg.use_pattern_utility_gate,
            )
            if is_utility_safe:
                adapter, route_reason = "factual", "utility_safe_override"
        model.set_adapter(adapter)
        completion = generate_completion(
            model,
            tokenizer,
            build_prompt(tokenizer, prompt, cfg.use_chat_template),
            cfg.max_new_tokens,
        )
        postprocess_action = "none"
        if (
            cfg.use_safe_feature_recovery
            and adapter == "factual"
            and route_reason == "semantic_benign_gate"
            and classify_refusal(completion)
        ):
            completion = SAFE_RECOVERY_TEXT
            postprocess_action = "semantic_benign_recovery"
        elif cfg.use_fixed_high_risk_refusal and adapter == "safety" and route_reason == "risk_override" and not classify_refusal(completion):
            completion = ROUTER_REFUSAL_TEXT
            postprocess_action = "fixed_high_risk_refusal"
        elif (
            cfg.use_safe_feature_recovery
            and adapter == "factual"
            and classify_refusal(completion)
            and should_recover_safe_feature_refusal(prompt, flags, cfg.benign_gate_min, cfg.safe_recovery_mode)
        ):
            completion = SAFE_RECOVERY_TEXT
            postprocess_action = "safe_feature_recovery"
        elif (
            cfg.use_low_risk_factual_recovery
            and adapter == "factual"
            and route_reason in {"calibrator", "utility_safe_override", "benign_override"}
            and classify_refusal(completion)
            and is_low_risk_factual_context(
                flags,
                risk_score,
                route_probability,
                cfg.factual_recovery_max_risk,
                cfg.factual_recovery_max_route_probability,
            )
        ):
            completion = SAFE_RECOVERY_TEXT
            postprocess_action = "low_risk_factual_recovery"
        results.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "metadata": row["metadata"],
                "risk_score": risk_score,
                "ambiguity_score": ambiguity_score,
                "route_probability": route_probability,
                "feature_flags": flags,
                "adapter": adapter,
                "route_reason": route_reason,
                "utility_probability": utility_probability,
                "utility_gate_reason": utility_gate_reason,
                "postprocess_action": postprocess_action,
                "completion": completion,
                "is_refusal": classify_refusal(completion),
            }
        )

    summary = {
        "refusal_rate": sum(r["is_refusal"] for r in results) / len(results) if results else None,
        "safety_route_rate": sum(r["adapter"] == "safety" for r in results) / len(results) if results else None,
        "num_samples": len(results),
        "elapsed_seconds": time.time() - started,
        "fixed_high_risk_refusals": sum(1 for r in results if r["postprocess_action"] == "fixed_high_risk_refusal"),
        "safe_feature_recoveries": sum(1 for r in results if r["postprocess_action"] == "safe_feature_recovery"),
        "semantic_benign_recoveries": sum(
            1 for r in results if r["postprocess_action"] == "semantic_benign_recovery"
        ),
        "low_risk_factual_recoveries": sum(
            1 for r in results if r["postprocess_action"] == "low_risk_factual_recovery"
        ),
    }
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg.__dict__,
                "router_v7_metadata": v7_bundle.get("metadata"),
                "router_v7_metrics": v7_bundle.get("metrics"),
                "summary": summary,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved results to {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
