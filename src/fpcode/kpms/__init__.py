"""Lazy public exports; importing the package initializes no optional backend."""

from importlib import import_module

_EXPORTS = {
    "IncompatibleModelsError": ("evaluation", "IncompatibleModelsError"),
    "invert_permutation": ("evaluation", "invert_permutation"),
    "is_permutation": ("evaluation", "is_permutation"),
    "mapping_hash": ("evaluation", "mapping_hash"),
    "old_to_new_map": ("evaluation", "old_to_new_map"),
    "read_checkpoint_identity": ("evaluation", "read_checkpoint_identity"),
    "reconstruct_mask": ("evaluation", "reconstruct_mask"),
    "relabel_states": ("evaluation", "relabel_states"),
    "stable_frequency_order": ("evaluation", "stable_frequency_order"),
    "validate_model_compatibility": ("evaluation", "validate_model_compatibility"),
    "extract_syllable_statistics": ("metrics", "extract_syllable_statistics"),
    "get_frequencies": ("metrics", "get_frequencies"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(f".{module_name}", __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
