from datasets import load_dataset
from src.config import DatasetConfig


DATASET_MAP = {
    "contract-nli": "kiddothe2b/contract-nli",
    "stereoset": "McGill-NLP/stereoset",
}

SUBSET_MAP = {
    "contract-nli": "contractnli_a",
    "stereoset": "intersentence",
}

LABEL_MAPS = {
    "contract-nli": {0: "Contradiction", 1: "Entailment", 2: "NotMentioned"},
    "stereoset": ["gender", "race", "profession", "religion"],
}


def _flatten_stereoset(raw: dict) -> dict:
    # sentences.gold_label: 0 = anti-stereotype, 1 = stereotype, 2 = unrelated
    sents = raw.get("sentences", {})
    by_label = dict(zip(sents.get("gold_label", []), sents.get("sentence", [])))
    return {
        "context": raw.get("context", ""),
        "target": raw.get("target", ""),
        "category": raw.get("bias_type", ""),
        "stereo_sentence": by_label.get(1, ""),
        "unstereo_sentence": by_label.get(0, ""),
    }


class DatasetExample:
    def __init__(self, raw: dict, dataset_name: str):
        self._raw = raw
        self._name = dataset_name

    def get(self, key, default=None):
        return self._raw.get(key, default)

    def __getattr__(self, key):
        return self._raw.get(key)

    def _get_input_text(self) -> str:
        if self._name == "contract-nli":
            return self._raw.get("premise", "")
        return self._raw.get("context", "")

    def _get_hypothesis(self) -> str:
        if self._name == "contract-nli":
            return self._raw.get("hypothesis", "")
        return self._raw.get("stereo_sentence", "")

    def _get_gold_label(self) -> str:
        if self._name == "contract-nli":
            label = self._raw.get("label", "")
            if isinstance(label, (int, float)):
                return LABEL_MAPS["contract-nli"].get(int(label), str(label))
            return str(label)
        return str(self._raw.get("category", ""))


class DatasetWrapper:
    def __init__(self, examples: list[DatasetExample], dataset_name: str):
        self._examples = examples
        self._name = dataset_name

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx) -> DatasetExample:
        return self._examples[idx]

    def __iter__(self):
        return iter(self._examples)


def get_labels(dataset_name: str) -> list[str]:
    if dataset_name == "contract-nli":
        return list(LABEL_MAPS["contract-nli"].values())
    return LABEL_MAPS.get("stereoset", [])


def load_dataset_by_config(
    config: DatasetConfig,
    split: str | None = None,
    max_samples: int | None = None,
) -> DatasetWrapper:
    hf_id = config.huggingface_id or DATASET_MAP.get(config.name)
    if not hf_id:
        raise ValueError(
            f"Unknown dataset '{config.name}'. "
            f"Supported: {list(DATASET_MAP.keys())}. "
            f"Or set huggingface_id in config."
        )

    dataset = load_dataset(
        hf_id,
        revision="refs/convert/parquet",
        data_dir=SUBSET_MAP.get(config.name),
    )
    ds = dataset[split or config.split]

    max_samples = max_samples if max_samples is not None else config.max_samples
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    if config.name == "stereoset":
        examples = [DatasetExample(_flatten_stereoset(ex), config.name) for ex in ds]
    else:
        examples = [DatasetExample(ex, config.name) for ex in ds]
    return DatasetWrapper(examples, config.name)
