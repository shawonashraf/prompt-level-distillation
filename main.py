import argparse
import dataclasses
import json
import logging
import os
import sys

import wandb

from src.config import Config, load_config
from src.data import load_dataset_by_config, get_labels
from src.models import ExtractionResult, ClusterResult
from src.phase1_extract import extract_instructions, save_instructions
from src.phase2_cluster import cluster_and_synthesize, save_clusters
from src.phase3_conflict import resolve_conflicts, save_conflict_result
from src.phase4_inference import run_inference
from src.evaluate import compute_macro_f1
from src.utils import get_stats

log = logging.getLogger("pld")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(description="Prompt-Level Distillation")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["extract", "cluster", "conflict", "evaluate", "all"],
        default=None,
        help="Run only a specific phase (overrides config)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = load_config(args.config)

    if args.phase:
        if args.phase == "all":
            config.phases = type(config.phases)(
                extract=True, cluster=True, conflict=True, evaluate=True
            )
        else:
            for attr in ["extract", "cluster", "conflict", "evaluate"]:
                setattr(config.phases, attr, attr == args.phase)

    os.makedirs(config.output_dir, exist_ok=True)

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        wandb.login(key=api_key)

    wandb.init(
        project=config.wandb.project,
        name=config.wandb.run_name,
        entity=config.wandb.entity,
        config=dataclasses.asdict(config),
    )

    log.info(f"Config: {args.config}")
    log.info(f"Dataset: {config.dataset.name}, split: {config.dataset.split}")
    log.info(f"Teacher: {config.teacher.model}")
    log.info(f"Student: {config.student.model}")

    dataset = load_dataset_by_config(config.dataset)
    labels = get_labels(config.dataset.name)
    log.info(f"Loaded {len(dataset)} examples, labels: {labels}")

    consolidated_instructions = []

    if config.phases.extract:
        log.info("=" * 60)
        log.info("PHASE 1: Supervised Instruction Extraction")
        log.info("=" * 60)

        extractions = extract_instructions(config, dataset)
        extract_path = save_instructions(extractions, config.output_dir)

        success_count = sum(1 for e in extractions if e.success)
        log.info(
            f"Extraction complete: {success_count}/{len(extractions)} successful"
        )

        wandb.log({
            "phase1/total_extractions": len(extractions),
            "phase1/successful": success_count,
            "phase1/failure_rate": 1 - (success_count / len(extractions)),
        })

        table = wandb.Table(columns=["index", "gold_label", "rule", "success"])
        for e in extractions[:50]:
            table.add_data(
                e.index, e.gold_label, e.executable_rule[:200], e.success
            )
        wandb.log({"phase1/sample_instructions": table})

    if config.phases.cluster:
        log.info("=" * 60)
        log.info("PHASE 2: Clustering Logic Synthesis")
        log.info("=" * 60)

        if not config.phases.extract:
            with open(os.path.join(config.output_dir, "extracted_instructions.json")) as f:
                extractions = [ExtractionResult(**r) for r in json.load(f)]
        clusters = cluster_and_synthesize(config, extractions)

        cluster_path = save_clusters(clusters, config.output_dir)
        consolidated_instructions = [
            c.consolidated_instruction
            for c in clusters
            if c.consolidated_instruction
        ]

        log.info(f"Clustering complete: {len(clusters)} clusters")

        wandb.log({
            "phase2/num_clusters": len(clusters),
            "phase2/avg_rules_per_cluster": (
                sum(len(c.member_rules) for c in clusters) / len(clusters)
                if clusters
                else 0
            ),
        })

        table = wandb.Table(columns=["cluster_id", "size", "consolidated_instruction"])
        for c in clusters:
            table.add_data(
                c.cluster_id,
                len(c.member_rules),
                c.consolidated_instruction[:300],
            )
        wandb.log({"phase2/clusters": table})

    if config.phases.conflict:
        log.info("=" * 60)
        log.info("PHASE 3: Conflict Resolution")
        log.info("=" * 60)

        if not config.phases.cluster:
            with open(os.path.join(config.output_dir, "clusters.json")) as f:
                clusters = [ClusterResult(**c) for c in json.load(f)]

        conflict_result = resolve_conflicts(
            config,
            clusters,
            dataset,
            labels,
        )
        conflict_path = save_conflict_result(conflict_result, config.output_dir)
        consolidated_instructions = conflict_result.consolidated_instructions

        log.info(
            f"Conflict resolution complete: {conflict_result.iteration} iterations, "
            f"F1={conflict_result.current_f1:.4f}, converged={conflict_result.converged}"
        )

        wandb.log({
            "phase3/iterations": conflict_result.iteration,
            "phase3/f1_before": conflict_result.previous_f1,
            "phase3/f1_after": conflict_result.current_f1,
            "phase3/converged": conflict_result.converged,
        })

    if config.phases.evaluate:
        log.info("=" * 60)
        log.info("PHASE 4: Final Evaluation")
        log.info("=" * 60)

        if not consolidated_instructions:
            with open(os.path.join(config.output_dir, "conflict_resolution.json")) as f:
                cr = json.load(f)
            consolidated_instructions = cr.get("consolidated_instructions", [])

        if not consolidated_instructions:
            with open(os.path.join(config.output_dir, "clusters.json")) as f:
                cl = json.load(f)
            consolidated_instructions = [
                c["consolidated_instruction"] for c in cl if c["consolidated_instruction"]
            ]

        if config.dataset.eval_split and config.dataset.eval_split != config.dataset.split:
            eval_dataset = load_dataset_by_config(
                config.dataset,
                split=config.dataset.eval_split,
                max_samples=config.dataset.eval_max_samples or config.dataset.max_samples,
            )
            log.info(
                f"Evaluating on held-out split '{config.dataset.eval_split}' "
                f"({len(eval_dataset)} examples)"
            )
        else:
            eval_dataset = dataset
            log.warning(
                "No held-out eval_split configured; evaluating on the same "
                "data used for distillation (train leakage)."
            )

        predictions = run_inference(
            config, eval_dataset, consolidated_instructions, labels
        )

        gold_labels = [ex._get_gold_label() for ex in eval_dataset]
        f1 = compute_macro_f1(gold_labels, predictions, labels)

        log.info(f"Final Macro F1: {f1:.4f}")

        wandb.log({
            "phase4/macro_f1": f1,
            "phase4/num_instructions": len(consolidated_instructions),
        })

        with open(
            os.path.join(config.output_dir, "final_instructions.txt"), "w"
        ) as f:
            for i, inst in enumerate(consolidated_instructions, 1):
                f.write(f"{i}. {inst}\n\n")

        log.info(f"Saved final instructions to {config.output_dir}/final_instructions.txt")

    stats = get_stats()
    wandb.log({
        "llm/requests": stats.requests,
        "llm/input_tokens": stats.input_tokens,
        "llm/output_tokens": stats.output_tokens,
        "llm/reasoning_tokens": stats.reasoning_tokens,
    })
    log.info(f"LLM stats: {stats.requests} requests, {stats.input_tokens} input tokens, {stats.output_tokens} output tokens, {stats.reasoning_tokens} reasoning tokens")

    wandb.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
