from dataclasses import dataclass, field
from typing import Optional
import yaml
import os


@dataclass
class ModelConfig:
    model: str
    base_url: str = "http://localhost:1234/v1"
    temperature: float = 0.7
    # null = no client-side cap; the server allows up to the model's max context
    max_tokens: Optional[int] = 4096
    api_key: str = ""
    # parallel requests; vLLM batches them, LM Studio should stay at 1
    concurrency: int = 1


@dataclass
class DatasetConfig:
    name: str = "contract-nli"
    split: str = "train"
    max_samples: Optional[int] = None
    huggingface_id: Optional[str] = None
    # held-out split for Phase 4 evaluation; None falls back to `split` (leakage)
    eval_split: Optional[str] = None
    eval_max_samples: Optional[int] = None
    # dataset saved via datasets.save_to_disk; bypasses hub resolution entirely
    # (offline compute nodes choke on data_dir cache-key lookups)
    local_path: Optional[str] = None


@dataclass
class ClusteringConfig:
    # kmeans: MiniLM embeddings of extracted rules form one dense continuum,
    # so DBSCAN chains everything into a single cluster at any workable eps
    # (measured on 6,819 contract-nli rules)
    algorithm: str = "dbscan"
    n_clusters: int = 17
    eps: float = 0.5
    min_samples: int = 2
    embedding_model: str = "all-MiniLM-L6-v2"
    # cap rules included in one synthesis prompt; a 6,800-rule cluster
    # overflowed the teacher's context
    max_rules_per_synthesis: int = 150


@dataclass
class ConflictResolutionConfig:
    max_iterations: int = 5
    convergence_threshold: float = 0.01
    sample_size: int = 20


@dataclass
class WandbConfig:
    project: str = "prompt-distill"
    run_name: Optional[str] = None
    entity: Optional[str] = None


@dataclass
class PhasesConfig:
    extract: bool = True
    cluster: bool = True
    conflict: bool = True
    evaluate: bool = True


@dataclass
class Config:
    teacher: ModelConfig
    student: ModelConfig
    dataset: DatasetConfig
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    conflict_resolution: ConflictResolutionConfig = field(
        default_factory=ConflictResolutionConfig
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    output_dir: str = "output"


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)

    teacher = ModelConfig(**data["teacher"])
    if not teacher.api_key:
        teacher.api_key = os.environ.get("OPENAI_API_KEY", "not-needed")

    student = ModelConfig(**data["student"])
    if not student.api_key:
        student.api_key = os.environ.get("OPENAI_API_KEY", "not-needed")

    dataset = DatasetConfig(**data.get("dataset", {}))
    clustering = ClusteringConfig(**data.get("clustering", {}))
    conflict = ConflictResolutionConfig(**data.get("conflict_resolution", {}))

    wandb_data = data.get("wandb", {})
    wandb = WandbConfig(**wandb_data)

    phases_data = data.get("phases", {})
    phases = PhasesConfig(**phases_data)

    return Config(
        teacher=teacher,
        student=student,
        dataset=dataset,
        clustering=clustering,
        conflict_resolution=conflict,
        wandb=wandb,
        phases=phases,
        output_dir=data.get("output_dir", "output"),
    )
