import argparse

from Thread_LlamaInvoker import LlamaInvoker


def build_config(args):
    return {
        "data_config": {
            "input_filepath": args.input_filepath,
            "output_filepath": args.output_filepath,
            "min_uid": args.min_uid,
            "max_uid": args.max_uid,
            "train_days": [args.train_days_start, args.train_days_end],
            "test_days": [args.test_days_start, args.test_days_end],
            "steps": args.steps,
            "task": args.task,
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build conversation-style mobility datasets.")
    parser.add_argument("--input_filepath", type=str, default="data/demo.csv.gz")
    parser.add_argument("--output_filepath", type=str, default="datasets/dataset_demo.json")
    parser.add_argument("--min_uid", type=int, default=0)
    parser.add_argument("--max_uid", type=int, default=10)
    parser.add_argument("--train_days_start", type=int, default=0)
    parser.add_argument("--train_days_end", type=int, default=59)
    parser.add_argument("--test_days_start", type=int, default=60)
    parser.add_argument("--test_days_end", type=int, default=74)
    parser.add_argument("--steps", type=int, default=None, help="Number of future steps for trajectory task.")
    parser.add_argument(
        "--task",
        type=str,
        default="trajectory",
        choices=["trajectory", "nextloc"],
        help="trajectory keeps original sequence targets; nextloc exports only first future step.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    invoker = LlamaInvoker(config)
    invoker.run()
