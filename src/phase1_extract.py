import json
import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from src.config import Config
from src.utils import chat_completion
from src.models import ExtractionResult

log = logging.getLogger(__name__)


def extract_instructions(config: Config, dataset) -> list[ExtractionResult]:
    env = Environment(loader=FileSystemLoader("prompts"))

    template_name = (
        "extract_contract_nli.j2"
        if config.dataset.name == "contract-nli"
        else "extract_stereoset.j2"
    )
    template = env.get_template(template_name)

    results = []
    total = len(dataset)

    for idx, example in enumerate(dataset):
        try:
            if config.dataset.name == "contract-nli":
                prompt = template.render(
                    contract_snippet=example.get("sentence1", ""),
                    hypothesis=example.get("sentence2", ""),
                    gold_label=example.get("label", ""),
                )
                input_text = example.get("sentence1", "")
                hypothesis = example.get("sentence2", "")
                gold_label = str(example.get("label", ""))
            else:
                prompt = template.render(
                    context=example.get("context", ""),
                    sentence_stem=example.get("sentence_stem", ""),
                    target=example.get("target", ""),
                    gold_category=example.get("category", ""),
                    stereo_sentence=example.get("stereo_sentence", ""),
                    unstereo_sentence=example.get("unstereo_sentence", ""),
                )
                input_text = example.get("context", "")
                hypothesis = example.get("sentence_stem", "")
                gold_label = str(example.get("category", ""))

            response = chat_completion(
                system="You are a reasoning teacher that extracts logic rules from examples.",
                user=prompt,
                config=config.teacher,
            )

            parsed = json.loads(response)
            results.append(
                ExtractionResult(
                    index=idx,
                    input_text=input_text,
                    hypothesis=hypothesis,
                    gold_label=gold_label,
                    reasoning_trace=parsed.get("reasoning_trace", ""),
                    executable_rule=parsed.get("executable_rule", ""),
                    success=True,
                )
            )
        except Exception as e:
            log.warning(f"Failed to extract instruction for index {idx}: {e}")
            results.append(
                ExtractionResult(
                    index=idx,
                    input_text=str(example),
                    hypothesis="",
                    gold_label="",
                    reasoning_trace="",
                    executable_rule="",
                    success=False,
                )
            )

        if (idx + 1) % 10 == 0 or idx == total - 1:
            log.info(f"Extracted {idx + 1}/{total} instructions")

    return results


def save_instructions(
    results: list[ExtractionResult], output_dir: str
) -> str:
    path = Path(output_dir) / "extracted_instructions.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "index": r.index,
            "input_text": r.input_text,
            "hypothesis": r.hypothesis,
            "gold_label": r.gold_label,
            "reasoning_trace": r.reasoning_trace,
            "executable_rule": r.executable_rule,
            "success": r.success,
        }
        for r in results
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved {len(results)} instructions to {path}")
    return str(path)
