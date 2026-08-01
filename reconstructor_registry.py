from reconstructor_algorithm_specs import (
    DISCOVERY_ALGORITHM_SPECS,
    EXPERIMENTAL_ALGORITHM_SPECS,
    LEGACY_ALGORITHM_SPECS,
    PUBLICATION_ALGORITHM_SPECS,
    build_algorithms,
)


LEGACY_ALGORITHM_NAMES = [spec.name for spec in LEGACY_ALGORITHM_SPECS]


def get_legacy_algorithms():
    return build_algorithms(LEGACY_ALGORITHM_SPECS)


def get_experimental_algorithms():
    return build_algorithms(EXPERIMENTAL_ALGORITHM_SPECS)


def get_publication_algorithms():
    return build_algorithms(PUBLICATION_ALGORITHM_SPECS)


def get_discovery_algorithms():
    return build_algorithms(DISCOVERY_ALGORITHM_SPECS)


def get_algorithms():
    return (
        get_legacy_algorithms()
        + get_experimental_algorithms()
        + get_publication_algorithms()
        + get_discovery_algorithms()
    )


def get_algorithm_map(algorithms=None):
    selected_algorithms = get_algorithms() if algorithms is None else algorithms
    return {
        getattr(algorithm, "__name__", str(algorithm)): algorithm
        for algorithm in selected_algorithms
    }


def resolve_reconstruction_algorithm(algorithm_name):
    if algorithm_name is None:
        return None
    if isinstance(algorithm_name, str) and algorithm_name.strip().lower() in {"", "none"}:
        return None

    algorithms_by_name = get_algorithm_map()
    if algorithm_name not in algorithms_by_name:
        available = ", ".join(sorted(algorithms_by_name))
        raise ValueError(
            f"Unknown reconstruction algorithm '{algorithm_name}'. Available options: {available}"
        )
    return algorithms_by_name[algorithm_name]


get_legacy_algorithms_to_test = get_legacy_algorithms
get_experimental_algorithms_to_test = get_experimental_algorithms
get_publication_algorithms_to_test = get_publication_algorithms
get_discovery_algorithms_to_test = get_discovery_algorithms


def get_algorithms_to_test():
    """Return algorithms safe for the historical symmetric-distance runners."""
    return (
        get_legacy_algorithms()
        + get_experimental_algorithms()
        + get_publication_algorithms()
    )


__all__ = [
    "DISCOVERY_ALGORITHM_SPECS",
    "LEGACY_ALGORITHM_NAMES",
    "EXPERIMENTAL_ALGORITHM_SPECS",
    "LEGACY_ALGORITHM_SPECS",
    "PUBLICATION_ALGORITHM_SPECS",
    "get_algorithm_map",
    "get_algorithms",
    "get_algorithms_to_test",
    "get_discovery_algorithms",
    "get_discovery_algorithms_to_test",
    "get_experimental_algorithms",
    "get_experimental_algorithms_to_test",
    "get_legacy_algorithms",
    "get_legacy_algorithms_to_test",
    "get_publication_algorithms",
    "get_publication_algorithms_to_test",
    "resolve_reconstruction_algorithm",
]
