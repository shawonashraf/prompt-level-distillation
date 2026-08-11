import logging
from concurrent.futures import ThreadPoolExecutor

from jinja2 import Environment, FileSystemLoader

from src.config import Config
from src.utils import chat_completion

log = logging.getLogger(__name__)


def render_system_prompt(instructions: list[str], labels: list[str]) -> str:
    env = Environment(loader=FileSystemLoader("prompts"))
    return env.get_template("inference.j2").render(
        instructions=instructions,
        labels=labels,
    )


def run_inference(
    config: Config,
    dataset,
    instructions: list[str],
    labels: list[str],
) -> list[str]:
    system_prompt = render_system_prompt(instructions, labels)
    total = len(dataset)
    done = 0

    def infer_one(idx: int, example) -> str:
        nonlocal done
        try:
            input_text = example._get_input_text()
            hypothesis = example._get_hypothesis()
            user_prompt = f"Input: {input_text}\nHypothesis: {hypothesis}"

            response = chat_completion(
                system=system_prompt,
                user=user_prompt,
                config=config.student,
            )
            predicted = normalize_label(response.strip(), labels)
        except Exception as e:
            log.warning(f"Inference failed for index {idx}: {e}")
            predicted = ""

        done += 1
        if done % 50 == 0 or done == total:
            log.info(f"Inferred {done}/{total}")
        return predicted

    with ThreadPoolExecutor(max_workers=config.student.concurrency) as pool:
        futures = [
            pool.submit(infer_one, idx, ex) for idx, ex in enumerate(dataset)
        ]
        return [f.result() for f in futures]


def normalize_label(predicted: str, labels: list[str]) -> str:
    predicted_lower = predicted.lower().strip()
    if not predicted_lower:
        return ""
    best = ""
    best_len = len(predicted_lower)
    for label in labels:
        label_lower = label.lower()
        idx = predicted_lower.find(label_lower)
        if idx != -1:
            # ponytail: prefer the earliest (leftmost) match to avoid
            # "NotMentioned" being shadowed by substrings
            if idx < best_len:
                best_len = idx
                best = label
    return best if best else predicted.strip()
