import json
import logging
from pathlib import Path
from collections import defaultdict

import wandb
from jinja2 import Environment, FileSystemLoader

from src.config import Config
from src.utils import chat_completion, parse_json_response
from src.models import ClusterResult, ConflictResult
from src.evaluate import compute_macro_f1
from src.phase4_inference import run_inference

log = logging.getLogger(__name__)


def resolve_conflicts(
    config: Config,
    clusters: list[ClusterResult],
    dataset,
    labels: list[str],
) -> ConflictResult:
    current_instructions = [
        c.consolidated_instruction for c in clusters if c.consolidated_instruction
    ]
    if not current_instructions:
        log.warning("No consolidated instructions for conflict resolution.")
        return ConflictResult(
            iteration=0,
            previous_f1=0.0,
            current_f1=0.0,
            consolidated_instructions=[],
            converged=True,
            error_examples=[],
        )

    env = Environment(loader=FileSystemLoader("prompts"))
    conflict_template = env.get_template("conflict_resolution.j2")

    prev_f1 = 0.0
    best_instructions = current_instructions

    for iteration in range(config.conflict_resolution.max_iterations):
        log.info(f"Conflict resolution iteration {iteration + 1}")

        predictions = run_inference(
            config,
            dataset,
            best_instructions,
            labels,
        )

        current_f1 = compute_macro_f1(
            [ex._get_gold_label() for ex in dataset],
            predictions,
            labels,
        )

        log.info(
            f"  Iteration {iteration + 1}: F1 = {current_f1:.4f} "
            f"(prev = {prev_f1:.4f})"
        )
        wandb.log({
            "phase3/iteration": iteration + 1,
            "phase3/iter_f1": current_f1,
            "phase3/num_instructions": len(best_instructions),
        })

        if abs(current_f1 - prev_f1) <= config.conflict_resolution.convergence_threshold:
            log.info(f"  Converged at iteration {iteration + 1}")
            return ConflictResult(
                iteration=iteration + 1,
                previous_f1=prev_f1,
                current_f1=current_f1,
                consolidated_instructions=best_instructions,
                converged=True,
                error_examples=[],
            )

        errors = []
        successes = []
        for i, pred in enumerate(predictions):
            if i >= len(dataset):
                break
            example = dataset[i]
            gold = example._get_gold_label()
            if pred != gold:
                errors.append({
                    "input_text": example._get_input_text(),
                    "hypothesis": example._get_hypothesis(),
                    "gold_label": gold,
                    "predicted_label": pred,
                })
            else:
                successes.append({
                    "input_text": example._get_input_text(),
                    "hypothesis": example._get_hypothesis(),
                    "gold_label": gold,
                })

        sample_errors = errors[: config.conflict_resolution.sample_size]
        sample_successes = successes[: config.conflict_resolution.sample_size]

        if not sample_errors:
            log.info("  No errors found. Stopping.")
            return ConflictResult(
                iteration=iteration + 1,
                previous_f1=prev_f1,
                current_f1=current_f1,
                consolidated_instructions=best_instructions,
                converged=True,
                error_examples=[],
            )

        prompt = conflict_template.render(
            instructions=best_instructions,
            failures=sample_errors,
            successes=sample_successes,
        )

        response = chat_completion(
            system="You refine reasoning instructions based on error analysis.",
            user=prompt,
            config=config.teacher,
        )

        parsed = parse_json_response(response)
        best_instructions = parsed.get("refined_instructions", best_instructions)
        prev_f1 = current_f1

    return ConflictResult(
        iteration=config.conflict_resolution.max_iterations,
        previous_f1=prev_f1,
        current_f1=current_f1,
        consolidated_instructions=best_instructions,
        converged=False,
        error_examples=sample_errors if sample_errors else [],
    )


def save_conflict_result(
    result: ConflictResult,
    output_dir: str,
) -> str:
    path = Path(output_dir) / "conflict_resolution.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "iteration": result.iteration,
        "previous_f1": result.previous_f1,
        "current_f1": result.current_f1,
        "consolidated_instructions": result.consolidated_instructions,
        "converged": result.converged,
        "error_examples": result.error_examples,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved conflict resolution result to {path}")
    return str(path)
