import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from src.config import Config
from src.utils import chat_completion, parse_json_response
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
    total = len(dataset)
    done = 0

    def extract_one(idx: int, example) -> ExtractionResult:
        nonlocal done
        try:
            input_text = example._get_input_text()
            hypothesis = example._get_hypothesis()
            gold_label = example._get_gold_label()

            if config.dataset.name == "contract-nli":
                prompt = template.render(
                    contract_snippet=input_text,
                    hypothesis=hypothesis,
                    gold_label=gold_label,
                )
            else:
                prompt = template.render(
                    context=input_text,
                    target=example.get("target", ""),
                    gold_category=gold_label,
                    stereo_sentence=example.get("stereo_sentence", ""),
                    unstereo_sentence=example.get("unstereo_sentence", ""),
                )

            response = chat_completion(
                system="You are a reasoning teacher that extracts logic rules from examples.",
                user=prompt,
                config=config.teacher,
            )

            parsed = parse_json_response(response)
            result = ExtractionResult(
                index=idx,
                input_text=input_text,
                hypothesis=hypothesis,
                gold_label=gold_label,
                reasoning_trace=parsed.get("reasoning_trace", ""),
                executable_rule=parsed.get("executable_rule", ""),
                success=True,
            )
        except Exception as e:
            log.warning(f"Failed to extract instruction for index {idx}: {e}")
            result = ExtractionResult(
                index=idx,
                input_text=str(example),
                hypothesis="",
                gold_label="",
                reasoning_trace="",
                executable_rule="",
                success=False,
            )

        done += 1
        if done % 10 == 0 or done == total:
            log.info(f"Extracted {done}/{total} instructions")
        return result

    with ThreadPoolExecutor(max_workers=config.teacher.concurrency) as pool:
        futures = [
            pool.submit(extract_one, idx, ex) for idx, ex in enumerate(dataset)
        ]
        return [f.result() for f in futures]


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
