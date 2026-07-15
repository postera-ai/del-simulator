import numpy as np

from del_simulator.core import LibraryWithAffinities


def test_pkd_only_derives_kd():
    """
    LibraryWithAffinities.__post_init__ should derive Kd from pKd when only pKd is
    supplied -- previously it checked `self.Kd is None`, but the dataclass default is an
    empty dict ({}), not None, so this derivation was dead code under normal keyword
    construction.
    """
    lib = LibraryWithAffinities(
        nsynthon_id=["a"], smiles=["CCO"], pKd={"x": 5.0, "y": 6.0}
    )

    assert lib.Kd == {"x": 10 ** (-5.0), "y": 10 ** (-6.0)}


def test_kd_only_derives_pkd():
    lib = LibraryWithAffinities(
        nsynthon_id=["a"], smiles=["CCO"], Kd={"x": 1e-5, "y": 1e-6}
    )

    assert lib.pKd["x"] == -np.log10(1e-5)
    assert lib.pKd["y"] == -np.log10(1e-6)


def test_neither_supplied_stays_empty_and_mutable():
    """
    Regression guard: scripts/selection_and_readout.py constructs LibraryWithAffinities
    with neither pKd nor Kd supplied, then assigns into both as plain dicts afterward
    (del_library.pKd[key] = ...). Both fields must stay {} (not None) so that pattern
    keeps working.
    """
    lib = LibraryWithAffinities(nsynthon_id=["a"], smiles=["CCO"])

    assert lib.pKd == {}
    assert lib.Kd == {}

    lib.pKd["z"] = 7.0
    lib.Kd["z"] = 1e-7
    assert lib.pKd == {"z": 7.0}
    assert lib.Kd == {"z": 1e-7}


def test_both_supplied_neither_overwritten():
    """
    When both pKd and Kd are supplied at construction (even if not numerically
    consistent with each other), neither should be overwritten by the cross-derivation.
    """
    lib = LibraryWithAffinities(
        nsynthon_id=["a"], smiles=["CCO"], pKd={"x": 5.0}, Kd={"x": 999.0}
    )

    assert lib.pKd == {"x": 5.0}
    assert lib.Kd == {"x": 999.0}
