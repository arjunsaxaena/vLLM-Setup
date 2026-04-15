import csv
import random
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .inference import VLLMInference
    from .prompts import SYSTEM_PROMPT
except ImportError:
    from inference import VLLMInference
    from prompts import SYSTEM_PROMPT


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


def load_random_rows(csv_path: Path, sample_size: int = 5) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return []

    size = min(sample_size, len(rows))
    return random.sample(rows, size)


def build_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser prompt:\n{user_prompt}\n"


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    return text.strip().replace(" ", "")


def main() -> None:
    csv_path = ROOT_DIR / "data" / "train.csv"
    sampled_rows = load_random_rows(csv_path=csv_path, sample_size=5)

    if not sampled_rows:
        print("No rows found in data/train.csv")
        return

    model = VLLMInference()
    print(f"Loaded {len(sampled_rows)} random row(s) from {csv_path}")

    boxed_count = 0
    correct_boxed_count = 0
    total_latency_ms = 0.0

    for idx, row in enumerate(sampled_rows, start=1):
        row_id = row.get("id", "").strip()
        prompt = row.get("prompt", "")
        expected_answer = row.get("answer", "").strip()

        result = model.generate(
            build_prompt(prompt),
            temperature=0.2,
            top_p=0.95,
            max_tokens=512,
        )
        response = result["response"].strip()
        total_latency_ms += float(result["elapsed_ms"])

        boxed_answer = extract_boxed_answer(response)
        has_boxed = boxed_answer is not None
        is_correct = (
            has_boxed
            and normalize_answer(boxed_answer) == normalize_answer(expected_answer)
        )

        if has_boxed:
            boxed_count += 1
        if is_correct:
            correct_boxed_count += 1

        print("\n" + "=" * 60)
        print(f"Sample {idx} | id={row_id}")
        print("Prompt:")
        print(prompt)
        print("\nModel Response:")
        print(response)
        print("\nChecks:")
        print(f"- Has \\boxed{{}} answer: {has_boxed}")
        if has_boxed:
            print(f"- Extracted boxed answer: {boxed_answer}")
            print(f"- Expected answer: {expected_answer}")
            print(f"- Boxed answer correct: {is_correct}")
        print(f"\nLatency: {result['elapsed_ms']} ms")

    total = len(sampled_rows)
    boxed_rate = (boxed_count / total) * 100
    boxed_accuracy_overall = (correct_boxed_count / total) * 100
    boxed_accuracy_conditional = (
        (correct_boxed_count / boxed_count) * 100 if boxed_count else 0.0
    )
    avg_latency_ms = total_latency_ms / total

    print("\n" + "#" * 60)
    print("Evaluation Metrics")
    print(f"- Total samples: {total}")
    print(f"- Responses with \\boxed{{}}: {boxed_count}/{total} ({boxed_rate:.2f}%)")
    print(
        "- Correct boxed answers (overall): "
        f"{correct_boxed_count}/{total} ({boxed_accuracy_overall:.2f}%)"
    )
    print(
        "- Correct boxed answers (when boxed exists): "
        f"{correct_boxed_count}/{boxed_count if boxed_count else 0} "
        f"({boxed_accuracy_conditional:.2f}%)"
    )
    print(f"- Average latency: {avg_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
