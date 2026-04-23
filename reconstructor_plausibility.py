import numpy as np


def is_biologically_plausible_ancestor(ancestor, descendant):
    """
    Return True if ancestor could biologically be a parent of descendant.

    Current constraint: a lineage cannot regain a positive copy number at a
    locus where the ancestor already has copy number 0.
    """
    return not np.any((ancestor.genome == 0) & (descendant.genome > 0))


def is_biologically_plausible_pair(x, y):
    return (
        is_biologically_plausible_ancestor(x, y)
        or is_biologically_plausible_ancestor(y, x)
    )


_is_biologically_plausible_ancestor = is_biologically_plausible_ancestor
_is_biologically_plausible_pair = is_biologically_plausible_pair


__all__ = [
    "_is_biologically_plausible_ancestor",
    "_is_biologically_plausible_pair",
    "is_biologically_plausible_ancestor",
    "is_biologically_plausible_pair",
]
