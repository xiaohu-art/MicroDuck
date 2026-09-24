import re
from collections.abc import Sequence


def resolve_matching_names(
    patterns: str | Sequence[str],
    names: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Select entries of ``names`` that match the given regex patterns.

    Args:
        patterns: A regex pattern or a sequence of regex patterns.
        names: The candidate names, e.g. all joint names of the robot.
        preserve_order: If False (default), results follow the order of
            ``names``. If True, results follow the order of ``patterns``
            (names matched by the same pattern keep their order in ``names``).

    Returns:
        A tuple ``(indices, matched_names)``, where ``indices`` are positions
        in ``names``.

    Raises:
        ValueError: If a name is matched by more than one pattern, or if a
            pattern does not match any name.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns = list(patterns)
    names = list(names)

    # Each hit is (pattern_index, name_index, name).
    hits: list[tuple[int, int, str]] = []
    for name_idx, name in enumerate(names):
        matched = [k for k, p in enumerate(patterns) if re.fullmatch(p, name)]
        if len(matched) > 1:
            raise ValueError(
                f"Name '{name}' is matched by multiple patterns: "
                f"{[patterns[k] for k in matched]}"
            )
        if matched:
            hits.append((matched[0], name_idx, name))

    used = {pattern_idx for pattern_idx, _, _ in hits}
    unused = [p for k, p in enumerate(patterns) if k not in used]
    if unused:
        raise ValueError(
            f"Patterns did not match any name: {unused}\n"
            f"Available names: {names}"
        )

    if preserve_order:
        hits.sort(key=lambda h: (h[0], h[1]))

    return [h[1] for h in hits], [h[2] for h in hits]