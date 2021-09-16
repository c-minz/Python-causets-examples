#!/usr/bin/env python
'''
Created on 22 Aug 2020

This is a script to compute the cardinalities of the 1- and 2-layer past 
infinities for sprinkles in an Alexandrov subset of 1 + 1 Minkowski space.
The values are determined in two ways: simulated and analytic.

@author: Christoph Minz
'''
from __future__ import annotations
from typing import Tuple, List
from causets import Causet, HarmonicNumbers
from math import prod, factorial
import numpy as np
from numpy.random import default_rng

N: List[int] = [1]  # [1, 2, 5, 10, 15], [50, 100, 200]

iter_k_max: int = 32
iter_n_max: int = 256
H: np.ndarray = HarmonicNumbers(max(iter_n_max, max(N)))


def analytic_canonicalC1(n: int) -> float:
    print(f'... computing C_1 for n={n} ... ')
    return H[n] / n


def analytic_canonicalC2(n: int) -> float:
    print(f'... computing C_2 for n={n} ... ')
    return sum((H[n] - H[k]) / factorial(k)
               for k in range(min(n, iter_k_max))) / n


def simulated_canonical(n: int) -> Tuple[float, ...]:
    i_max: int = 10
    j_max: int = 10000
    canonicalC1: List[float] = [0.] * i_max
    canonicalC2: List[float] = [0.] * i_max
    for i in range(i_max):
        print(f'... simulating for n={n}, ' +
              f'iteration {i}/{i_max} ... ', end='')
        tempC1: int = 0
        tempC2: int = 0
        for j in range(j_max):
            C: Causet = Causet.FromPermutation(np.random.permutation(n))
            tempC1 += C.PastInfCard
            tempC2 += len(C.Layers(C.PastInf, 0, 1))
        canonicalC1[i] = float(tempC1) / j_max / n
        canonicalC2[i] = float(tempC2) / j_max / n
        print(f'C_1: {canonicalC1[i]:10.6}, C_2: {canonicalC2[i]:10.6}')
    return (sum(canonicalC1) / i_max, sum(canonicalC2) / i_max)


def analytic_grandcanonicalC1(lam: int) -> float:
    print(f'... computing C_1 for rho={lam}/a^2 ... ')
    return np.exp(-lam) / lam * sum(prod(lam / k
                                         for k in range(1, n + 1)) * H[n]
                                    for n in range(iter_n_max - 1, - 1, -1))


def analytic_grandcanonicalC2(lam: int) -> float:
    print(f'... computing C_2 for rho={lam}/a^2 ... ')
    return np.exp(-lam) / lam * sum(prod(lam / k for k in range(1, n + 1)) *
                                    sum((H[n] - H[k]) / factorial(k)
                                        for k in range(min(n, iter_k_max)))
                                    for n in range(iter_n_max - 1, - 1, -1))


def simulated_grandcanonical(lam: int) -> Tuple[float, ...]:
    rng = default_rng()
    i_max: int = 10
    j_max: int = 10000
    canonicalC1: List[float] = [0.] * i_max
    canonicalC2: List[float] = [0.] * i_max
    for i in range(i_max):
        print(f'... simulating for rho={lam}/a^2, ' +
              f'iteration {i}/{i_max} ... ', end='')
        tempC1: float = 0.
        tempC2: float = 0.
        for j in range(j_max):
            c: int = int(rng.poisson(lam=lam))
            if c > 0:
                C: Causet = Causet.FromPermutation(np.random.permutation(c))
                tempC1 += float(C.PastInfCard)
                tempC2 += float(len(C.Layers(C.PastInf, 0, 1)))
        canonicalC1[i] = tempC1 / j_max / lam
        canonicalC2[i] = tempC2 / j_max / lam
        print(f'C_1: {canonicalC1[i]:10.6}, C_2: {canonicalC2[i]:10.6}')
    return (sum(canonicalC1) / i_max, sum(canonicalC2) / i_max)


print('Computing relative cardinality for past infinites ...')
ana_canC1: List[Tuple[float, ...]] = [analytic_canonicalC1(n) for n in N]
ana_canC2: List[Tuple[float, ...]] = [analytic_canonicalC2(n) for n in N]
ana_grcC1: List[Tuple[float, ...]] = [analytic_grandcanonicalC1(l) for l in N]
ana_grcC2: List[Tuple[float, ...]] = [analytic_grandcanonicalC2(l) for l in N]
print('Simulating relative cardinality for past infinites ...')
sim_can: List[Tuple[float, ...]] = [simulated_canonical(n) for n in N]
sim_grc: List[Tuple[float, ...]] = [simulated_grandcanonical(n) for n in N]

# print result table:
print()
print('##### Canonical results #####')
print(' n =            ', end='')
for n in N:
    print('{:>10}'.format(n), end='')
print()
print('C_1, analytic   ', end='')
for v in ana_canC1:
    print('{:10.6}'.format(v), end='')
print()
print('C_1, simulated  ', end='')
for values in sim_can:
    print('{:10.6}'.format(values[0]), end='')
print()
print('C_2, analytic   ', end='')
for v in ana_canC2:
    print('{:10.6}'.format(v), end='')
print()
print('C_2, simulated  ', end='')
for values in sim_can:
    print('{:10.6}'.format(values[1]), end='')
print()
print()
print('##### Grand-canonical results #####')
print(' rho / a^-2 =   ', end='')
for n in N:
    print('{:>10}'.format(n), end='')
print()
print('C_1, analytic   ', end='')
for v in ana_grcC1:
    print('{:10.6}'.format(v), end='')
print()
print('C_1, simulated  ', end='')
for values in sim_grc:
    print('{:10.6}'.format(values[0]), end='')
print()
print('C_2, analytic   ', end='')
for v in ana_grcC2:
    print('{:10.6}'.format(v), end='')
print()
print('C_2, simulated  ', end='')
for values in sim_grc:
    print('{:10.6}'.format(values[1]), end='')
