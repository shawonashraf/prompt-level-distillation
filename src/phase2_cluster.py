import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader

from src.config import Config
from src.utils import chat_completion, parse_json_response
from src.models import ExtractionResult, ClusterResult

log = logging.getLogger(__name__)


def cluster_and_synthesize(
    config: Config,
    extractions: list[ExtractionResult],
) -> list[ClusterResult]:
    successful = [e for e in extractions if e.success and e.executable_rule]
    if not successful:
        log.warning("No successful extractions to cluster.")
        return []

    rules = [e.executable_rule for e in successful]
    indices = [e.index for e in successful]

    log.info(f"Loading embedding model: {config.clustering.embedding_model}")
    embedder = SentenceTransformer(config.clustering.embedding_model)
    embeddings = embedder.encode(rules, show_progress_bar=True)

    if config.clustering.algorithm == "kmeans":
        norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        k = min(config.clustering.n_clusters, len(rules))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(norm)
        log.info(f"KMeans produced {k} clusters")
    else:
        db = DBSCAN(
            eps=config.clustering.eps,
            min_samples=config.clustering.min_samples,
            metric="cosine",
        )
        labels = db.fit_predict(embeddings)

        unique_labels = set(labels)
        noise_count = 1 if -1 in unique_labels else 0
        n_clusters = len(unique_labels) - noise_count
        log.info(
            f"DBSCAN found {n_clusters} clusters, {noise_count} noise points "
            f"(eps={config.clustering.eps}, min_samples={config.clustering.min_samples})"
        )

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = {"indices": [], "rules": []}
        clusters[label]["indices"].append(indices[idx])
        clusters[label]["rules"].append(rules[idx])

    env = Environment(loader=FileSystemLoader("prompts"))
    template = env.get_template("cluster_synthesis.j2")

    results = []
    for cluster_id, members in clusters.items():
        try:
            cap = config.clustering.max_rules_per_synthesis
            prompt_rules = members["rules"]
            if len(prompt_rules) > cap:
                prompt_rules = random.Random(42).sample(prompt_rules, cap)
                log.info(
                    f"Cluster {cluster_id}: sampling {cap}/{len(members['rules'])} "
                    f"rules for the synthesis prompt"
                )
            prompt = template.render(rules=prompt_rules)
            response = chat_completion(
                system="You synthesize similar rules into unified instructions.",
                user=prompt,
                config=config.teacher,
            )

            parsed = parse_json_response(response)
            results.append(
                ClusterResult(
                    cluster_id=int(cluster_id),
                    member_indices=members["indices"],
                    member_rules=members["rules"],
                    consolidated_instruction=parsed.get(
                        "consolidated_instruction", ""
                    ),
                )
            )
            log.info(
                f"Cluster {cluster_id}: synthesized {len(members['rules'])} rules"
            )
        except Exception as e:
            log.error(f"Failed to synthesize cluster {cluster_id}: {e}")

    if clusters and not results:
        raise RuntimeError(
            f"All {len(clusters)} cluster syntheses failed; refusing to "
            f"continue with an empty instruction set."
        )

    return results


def save_clusters(
    results: list[ClusterResult],
    output_dir: str,
) -> str:
    path = Path(output_dir) / "clusters.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "cluster_id": r.cluster_id,
            "member_indices": r.member_indices,
            "member_rules": r.member_rules,
            "consolidated_instruction": r.consolidated_instruction,
        }
        for r in results
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved {len(results)} clusters to {path}")
    return str(path)
