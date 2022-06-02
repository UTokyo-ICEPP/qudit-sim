from typing import Optional
from types import ModuleType
import numpy as np

from rqutils import ArrayType
import rqutils.paulis as paulis
from rqutils.math import matrix_exp

def unitary_subtraction(
    time_evolution: ArrayType,
    heff_compos: ArrayType,
    offset_compos: ArrayType,
    tlist: ArrayType,
    basis_list: Optional[ArrayType] = None,
    npmod: ModuleType = np
) -> ArrayType:
    if isinstance(basis_list, type(None)):
        heff = paulis.compose(heff_compos, npmod=npmod)
        offset = paulis.compose(offset_compos, npmod=npmod)
    else:
        basis_list = basis_list.reshape(-1, *basis_list.shape[-2:])
        heff = npmod.tensordot(basis_list, heff_compos.reshape(-1), (0, 0))
        offset = npmod.tensordot(basis_list, offset_compos.reshape(-1), (0, 0))

    hermitian = heff[None, ...] * tlist[:, None, None]
    hermitian += offset
    unitary = matrix_exp(1.j * hermitian, hermitian=-1, npmod=npmod)

    return npmod.matmul(time_evolution, unitary)


def trace_norm_squared(unitary: ArrayType, npmod: ModuleType = np) -> ArrayType:
    norm_tr = npmod.trace(unitary, axis1=-2, axis2=-1) / unitary.shape[-1]
    return npmod.square(norm_tr.real) + npmod.square(norm_tr.imag)


def heff_fidelity(
    time_evolution: ArrayType,
    heff_compos: ArrayType,
    offset_compos: ArrayType,
    tlist: ArrayType,
    basis_list: Optional[ArrayType] = None,
    npmod: ModuleType = np
) -> ArrayType:
    target = unitary_subtraction(time_evolution, heff_compos, offset_compos, tlist,
                                 basis_list=basis_list, npmod=npmod)

    fidelity = trace_norm_squared(target, npmod=npmod)

    return fidelity
