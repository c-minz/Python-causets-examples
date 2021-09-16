#!/usr/bin/env python
'''
Created on 14 May 2021

@author: Christoph Minz
@license: BSD 3-Clause
'''
from typing import List
from causets.causet import Causet, CausetEvent
import causets.causetplotting as cplt
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

P: List[int]
#** 2D lattice with 7 x 7 events:
# P = [1, 8, 15, 22, 29, 36, 43,
#      2, 9, 16, 23, 30, 37, 44,
#      3, 10, 17, 24, 31, 38, 45,
#      4, 11, 18, 25, 32, 39, 46,
#      5, 12, 19, 26, 33, 40, 47,
#      6, 13, 20, 27, 34, 41, 48,
#      7, 14, 21, 28, 35, 42, 49])

#** 2D lattice with 8 x 8 events:
P = [1, 9, 17, 25, 33, 41, 49, 57,
     2, 10, 18, 26, 34, 42, 50, 58,
     3, 11, 19, 27, 35, 43, 51, 59,
     4, 12, 20, 28, 36, 44, 52, 60,
     5, 13, 21, 29, 37, 45, 53, 61,
     6, 14, 22, 30, 38, 46, 54, 62,
     7, 15, 23, 31, 39, 47, 55, 63,
     8, 16, 24, 32, 40, 48, 56, 64]

Cau = Causet.FromPermutation(P)
labelling: List[CausetEvent] = Cau.sortedByLabels()
C: np.ndarray = Cau.PastMatrix(labelling, float)
L: np.ndarray = Cau.LinkPastMatrix(labelling, float)
cplt.plotDiagram([Cau.find(i) for i in range(1, 65)], P)
plt.show()

# Pauli-Jordan operator as in [DSX17]:
E_DSX: np.ndarray = 0.5 * (C.T - C)
E_DSX_EV: np.ndarray = np.round(np.linalg.eigvals(E_DSX), decimals=6)
E_DSX_EV_Counter: Counter = Counter(E_DSX_EV.imag)
E_DSX_EV_Keys: np.ndarray = np.array(list(E_DSX_EV_Counter.keys()))
E_DSX_EV_Count: np.ndarray = np.array(list(E_DSX_EV_Counter.values()))
E_DSX_EV_Sorting: np.ndarray = np.argsort(E_DSX_EV_Keys)
E_DSX_EV_Keys = E_DSX_EV_Keys[E_DSX_EV_Sorting]
E_DSX_EV_Count = E_DSX_EV_Count[E_DSX_EV_Sorting]
plt.scatter(E_DSX_EV_Keys, E_DSX_EV_Count)
plt.show()
np.savetxt('PauliJordan_DSX_spec.csv',
           np.array([E_DSX_EV_Keys, E_DSX_EV_Count]).T, fmt='%.5f,%d')

# Pauli-Jordan operator as in [DFRW20]:
E_DFRW_pre: np.ndarray = 0.25 * (L + np.matmul(C, L))
E_DFRW: np.ndarray = (E_DFRW_pre.T - E_DFRW_pre)
E_DFRW_EV: np.ndarray = np.round(np.linalg.eigvals(E_DFRW), decimals=6)
E_DFRW_EV_Counter: Counter = Counter(E_DFRW_EV.imag)
E_DFRW_EV_Keys: np.ndarray = np.array(list(E_DFRW_EV_Counter.keys()))
E_DFRW_EV_Count: np.ndarray = np.array(list(E_DFRW_EV_Counter.values()))
E_DFRW_EV_Sorting: np.ndarray = np.argsort(E_DFRW_EV_Keys)
E_DFRW_EV_Keys = E_DFRW_EV_Keys[E_DFRW_EV_Sorting]
E_DFRW_EV_Count = E_DFRW_EV_Count[E_DFRW_EV_Sorting]
plt.scatter(E_DFRW_EV_Keys, E_DFRW_EV_Count)
plt.show()
np.savetxt('PauliJordan_DFRW_spec.csv',
           np.array([E_DFRW_EV_Keys, E_DFRW_EV_Count]).T, fmt='%.5f,%d')

#** References:
#
# [XDS17]
# X, Dowker, Surya.
# "Scalar Field Green Functions on Causal Sets"
# Classical and Quantum Gravity 34.12 (2017)
#
# [DFRW20]
# Dable-Heath, Fewster, Rejzner, Woods.
# "Algebraic Classical and Quantum Field Theory on Causal Sets"
# Physical Review D 101.6 (2020)
