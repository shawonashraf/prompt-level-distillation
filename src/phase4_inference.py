import logging

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

from src.config import Config
from src.utils import chat_completion

log = logging.getLogger(__name__)


def run_inference(
    client: OpenAI,
    config: Config,
    dataset,
    instructions: list[str],
    labels: list[str],
) -> list[str]:
    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template("inference.j2")

    predictions = []
    total = len(dataset)

    for idx, example in enumerate(dataset):
        try:
            input_text = example._get_input_text()
            hypothesis = example._get_hypothesis()

            system_prompt = template.render(
                instructions=instructions,
                labels=labels,
            )

            user_prompt = f"Input: {input_text}\nHypothesis: {hypothesis}"

            response = chat_completion(
                client,
                system=system_prompt,
                user=user_prompt,
                model=config.student.model,
                temperature=config.student.temperature,
                max_tokens=config.student.max_tokens,
            )

            predicted = response.strip()
            predicted = normalize_label(predicted, labels)
            predictions.append(predicted)
        except Exception as e:
            log.warning(f"Inference failed for index {idx}: {e}")
            predictions.append("")

        if (idx + 1) % 50 == 0 or idx == total - 1:
            log.info(f"Inferred {idx + 1}/{total}")

    return predictions


def normalize_label(predicted: str, labels: list[str]) -> str:
    predicted_lower = predicted.lower().strip()
    for label in labels:
        if label.lower() in predicted_lower:
            return label
    return predicted.strip()
