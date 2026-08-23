from __future__ import annotations

from typing import Any

from configs.config import Settings


def apply_overrides(config: Settings, overrides: dict[str, Any]) -> Settings:
    """
    Return a NEW Settings with dotted-path overrides applied on top of
    a deep copy of `config`. `config` itself is never mutated, so the
    same base Settings can be reused as the starting point for every
    variant in a sweep.

    Example:
        apply_overrides(base, {
            "embedding.provider": "sentence_transformers",
            "embedding.params.model": "intfloat/multilingual-e5-small",
        })

    Each dotted key walks/creates nested dicts down to the final
    segment, where the override value is set directly (so passing a
    dict value, e.g. {"chunking.params": {...}}, replaces that whole
    sub-dict rather than being merged key-by-key).
    """
    data = config.model_dump(mode="python")

    for dotted_key, value in overrides.items():
        _set_nested(data, dotted_key.split("."), value)

    return Settings.model_validate(data)


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = data

    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            # Overwrite non-dict / None / missing intermediate nodes
            # (e.g. an unset `retrieval.dense: None`) so nested
            # overrides can still be applied without pre-populating
            # every optional sub-block by hand.
            existing = {}
            cursor[key] = existing
        cursor = existing

    cursor[path[-1]] = value