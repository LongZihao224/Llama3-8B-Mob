import argparse
import csv
import glob
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional



def normalize_city(city: str) -> str:
    city = city.lower().strip()
    if city.startswith("city") and len(city) == 5:
        return city[-1]
    return city


def resolve_dataset_path(city: str, split: str, dataset_path: Optional[str]) -> str:
    if dataset_path:
        return dataset_path

    city = normalize_city(city)
    known = {
        "a": {
            "test": "datasets/datasetA_test_0-9999.json",
        },
        "b": {
            "test": "datasets/datasetB_test_22000-24999.json",
        },
        "c": {
            "test": "datasets/datasetC_test_17000-19999.json",
            "eval": "datasets/datasetC_eval_13600-16999.json",
        },
        "d": {
            "test": "datasets/datasetD_test_3000-5999.json",
            "eval": "datasets/datasetD_eval_2400-2999.json",
        },
    }
    if city in known and split in known[city] and Path(known[city][split]).exists():
        return known[city][split]

    candidates = sorted(glob.glob(f"datasets/dataset{city.upper()}_{split}_*.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Unable to resolve dataset for city={city}, split={split}. Pass --dataset_path explicitly."
    )


def parse_assistant_json(text: str) -> Dict:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].strip()
    if stripped.startswith("```"):
        stripped = stripped[len("```") :].strip()
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def to_location_id(record, grid_size: int) -> int:
    if isinstance(record, int):
        return int(record)
    if isinstance(record, list):
        if len(record) == 1:
            return int(record[0])
        if len(record) >= 4:
            x, y = int(record[-2]), int(record[-1])
            return x * grid_size + y
    raise ValueError(f"Unsupported record format for location id: {record}")


def extract_ground_truth(conversation: Dict, grid_size: int) -> int:
    assistant_messages = [m for m in conversation["messages"] if m["role"] == "assistant"]
    if not assistant_messages:
        raise ValueError("No assistant message found for ground truth")

    data = parse_assistant_json(assistant_messages[0]["content"])
    if "next_location_id" in data:
        gt = int(data["next_location_id"])
    elif "prediction" in data and isinstance(data["prediction"], list) and data["prediction"]:
        gt = to_location_id(data["prediction"][0], grid_size=grid_size)
    elif "next_record" in data:
        gt = to_location_id(data["next_record"], grid_size=grid_size)
    else:
        raise ValueError(f"Cannot extract next-location label from assistant content keys={list(data.keys())}")

    assert isinstance(gt, int), "ground truth must be int"
    return gt


def build_eval_messages(conversation: Dict, top_k: int, grid_size: int) -> List[Dict[str, str]]:
    messages = [m for m in conversation["messages"] if m["role"] != "assistant"]
    ranking_instruction = {
        "role": "user",
        "content": (
            f"\n\nNow solve the unified single-step next-location ranking task. "
            f"Predict the next location at t+1 only and return EXACTLY {top_k} candidate location IDs in descending confidence. "
            f"Each location ID is an integer in [0, {grid_size * grid_size - 1}] computed as x*{grid_size}+y. "
            f"Output format must be exactly: Predictions: id1, id2, ..., id{top_k}. "
            f"No JSON, no explanation, no extra text."
        ),
    }
    messages.append(ranking_instruction)
    return [{"from": m["role"], "value": m["content"]} for m in messages]


def parse_predictions(raw_text: str, top_k: int, max_valid_id: int) -> List[int]:
    nums = [int(x) for x in re.findall(r"-?\d+", raw_text)]
    parsed: List[int] = []
    seen = set()
    for n in nums:
        if n < 0 or n > max_valid_id:
            continue
        if n in seen:
            continue
        seen.add(n)
        parsed.append(n)
        if len(parsed) >= top_k:
            break
    assert all(isinstance(v, int) for v in parsed), "predictions must be list[int]"
    return parsed


def compute_metrics(preds: List[int], gt: int, top_k: int) -> Dict[str, float]:
    ranks = {pid: idx + 1 for idx, pid in enumerate(preds)}
    rank = ranks.get(gt)

    metrics = {
        "acc@1": 1.0 if gt in preds[:1] else 0.0,
        "acc@5": 1.0 if gt in preds[:5] else 0.0,
        "acc@10": 1.0 if gt in preds[:10] else 0.0,
        "mrr": 1.0 / rank if rank is not None else 0.0,
        "ndcg@10": 1.0 / math.log2(rank + 1) if (rank is not None and rank <= 10) else 0.0,
        "acc@k": 1.0 if gt in preds[:top_k] else 0.0,
    }
    return metrics


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def write_report(path: Path, report: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in report.items():
                if isinstance(v, (int, float, str)):
                    writer.writerow([k, v])
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Unified single-step next-location evaluation")
    parser.add_argument("--city", type=str, default="c", help="City abbreviation: a/b/c/d or cityB style")
    parser.add_argument("--split", type=str, default="test", help="Dataset split, e.g. test/eval")
    parser.add_argument("--k", type=int, default=10, help="Top-K predictions to evaluate")
    parser.add_argument("--ckpt", type=str, default="tangera/Llama3-8B-Mob", help="Model checkpoint path/name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional explicit dataset path")
    parser.add_argument("--out", type=str, default="results/nextloc_report.json", help="Output report path (.json or .csv)")
    parser.add_argument("--debug_out", type=str, default="results/nextloc_debug.json", help="Debug details output path")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of samples")
    parser.add_argument("--debug_limit", type=int, default=200, help="Number of samples stored in debug output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_path = resolve_dataset_path(args.city, args.split, args.dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None:
        data = data[: args.max_samples]

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    torch.manual_seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.ckpt,
        max_seq_length=50000,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    max_valid_id = args.grid_size * args.grid_size - 1
    sample_metrics = []
    debug_rows = []

    for idx, conversation in enumerate(data):
        gt = extract_ground_truth(conversation, grid_size=args.grid_size)
        assert 0 <= gt <= max_valid_id, f"gt out of range: {gt}"

        messages = build_eval_messages(conversation, top_k=args.k, grid_size=args.grid_size)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0,
            use_cache=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        raw_output = generated_text[len(prompt_text):].strip() if generated_text.startswith(prompt_text) else generated_text
        preds = parse_predictions(raw_output, top_k=args.k, max_valid_id=max_valid_id)
        assert isinstance(preds, list) and all(isinstance(v, int) for v in preds), "predictions are list[int]"

        metrics = compute_metrics(preds, gt, top_k=args.k)
        sample_metrics.append(metrics)

        if idx < args.debug_limit:
            debug_rows.append(
                {
                    "sample_index": idx,
                    "prompt": prompt_text,
                    "raw_model_output": raw_output,
                    "parsed_predictions": preds,
                    "ground_truth": gt,
                }
            )

    report = {
        "city": args.city,
        "split": args.split,
        "dataset_path": dataset_path,
        "checkpoint": args.ckpt,
        "top_k": args.k,
        "num_samples": len(sample_metrics),
        "acc@1": mean([m["acc@1"] for m in sample_metrics]),
        "acc@5": mean([m["acc@5"] for m in sample_metrics]),
        "acc@10": mean([m["acc@10"] for m in sample_metrics]),
        "mrr": mean([m["mrr"] for m in sample_metrics]),
        "ndcg@10": mean([m["ndcg@10"] for m in sample_metrics]),
        "acc@k": mean([m["acc@k"] for m in sample_metrics]),
    }

    write_report(Path(args.out), report)
    Path(args.debug_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.debug_out, "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Debug file saved to: {args.debug_out}")


if __name__ == "__main__":
    main()
