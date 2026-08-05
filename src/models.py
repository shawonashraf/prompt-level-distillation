from dataclasses import dataclass


@dataclass
class ExtractionResult:
    index: int
    input_text: str
    hypothesis: str
    gold_label: str
    reasoning_trace: str
    executable_rule: str
    success: bool


@dataclass
class ClusterResult:
    cluster_id: int
    member_indices: list[int]
    member_rules: list[str]
    consolidated_instruction: str


@dataclass
class ConflictResult:
    iteration: int
    previous_f1: float
    current_f1: float
    consolidated_instructions: list[str]
    converged: bool
    error_examples: list[dict]
