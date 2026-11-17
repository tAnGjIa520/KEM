"""Patch selection strategies used when storing MIL bags for replay.

Selector imports are lazy so lightweight buffer tools and their unit tests do
not eagerly import every optional MIL implementation.
"""

__all__ = ["SelectionResult", "build_replay_selector"]


def __getattr__(name):
    if name in __all__:
        from .selectors import SelectionResult, build_replay_selector
        return {"SelectionResult": SelectionResult, "build_replay_selector": build_replay_selector}[name]
    raise AttributeError(name)
