#!/usr/bin/env python
'''
Created on 25 Apr 2021

This file contains functions to get coordinates for animations of 
lightcones in causal sets.

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from math import sqrt

default_eps: float = 0.001


def move_centre(C: List[List[float]]) -> List[List[float]]:
    A: np.ndarray = np.array(C)
    Amin = np.min(A, axis=0)
    Amax = np.max(A, axis=0)
    Acen = (Amax + Amin) / 2
    for i in range(A.shape[0]):
        C[i] = (A[i, :] - Acen).tolist()
    return C


def get_1simplex(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 1-simplex.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    C: List[List[float]]
    s: float = edge / 2
    t: float = 0.75 * s * (1 + eps)
    if spacetime in {'black hole', 'Schwarzschild'}:
        C = [[-t, 0.47], [-t, 0.87], [t, 0.25]]
    else:
        C = [[-t, -s], [-t, s], [t, 0.]]
    return ('1-simplex', [2, 1, 3], C)


def get_2simplex(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 2-simplex.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    P: List[int] = [4, 2, 6, 1, 5, 3, 7]
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    r_h: float = r / 2  # radius half
    a: float = edge / 2
    tmax: float = a + r_h
    t: np.ndarray
    if spacetime in {'black hole', 'Schwarzschild'}:
        raise ValueError('2D black hole spacetime not supported')
    elif spacetime == 'de Sitter':
        t = (np.array([0, 1.5 * a, 1.5 * tmax]) - 0.75 * tmax) * (1 + eps)
    else:
        t = (np.array([0, a, tmax]) - tmax / 2) * (1 + eps)
    C: List[List[float]] = [[t[0], -a, -r_h],
                            [t[0], a, -r_h],
                            [t[1], 0., -r_h],
                            [t[0], 0., r],
                            [t[1], -a / 2, r_h / 2],
                            [t[1], a / 2, r_h / 2],
                            [t[2], 0., 0.]]
    return ('2-simplex', P, C)


def get_2simplexFlipped(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 2-simplex with one edge flipped.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    P: List[int] = [1, 5, 3, 6, 2, 4]
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    r_h: float = r / 2  # radius half
    a: float = edge / 2
    t: np.ndarray
    if spacetime in {'black hole', 'Schwarzschild'}:
        raise ValueError('2D black hole spacetime not supported')
    elif spacetime == 'de Sitter':
        t = np.array([-1.5 * a, 0, 1.5 * a]) * (1 + eps)
    else:
        t = np.array([-a, 0, a]) * (1 + eps)
    C: List[List[float]] = [[t[0], 0., -r_h],
                            [t[1], -a, -r_h],
                            [t[1], 0., r],
                            [t[2], -a / 2, r_h / 2],
                            [t[1], a, -r_h],
                            [t[2], a / 2, r_h / 2]]
    return ('2-simplex with one edge flipped', P, C)


def get_2simplexFlipped2(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 2-simplex with two edges flipped.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    P: List[int] = [3, 5, 1, 4, 2, 6]
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    r_h: float = r / 2  # radius half
    a: float = edge / 2
    t: np.ndarray
    if spacetime in {'black hole', 'Schwarzschild'}:
        raise ValueError('2D black hole spacetime not supported')
    elif spacetime == 'de Sitter':
        t = np.array([-1.5 * a, 0, 1.5 * a]) * (1 + eps)
    else:
        t = np.array([-a, 0, a]) * (1 + eps)
    C: List[List[float]] = [[t[0], -a / 2, r_h / 2],
                            [t[1], -a, -r_h],
                            [t[0], a / 2, r_h / 2],
                            [t[1], 0., r],
                            [t[1], a, -r_h],
                            [t[2], 0., -r_h]]
    return ('2-simplex with two edges flipped', P, C)


def get_2simplexRotated3(edge: float, s: int = 1, eps: float = default_eps,
                         spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 2-simplex rotating right/left-handed (3 steps).
    The rotation is right-handed if s == 1, it is left-handed if s == -1.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    P: List[int] = [6, 3, 5, 9, 7, 1, 8, 2, 4]
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    r_h: float = r / 2  # radius half
    a: float = edge / 2
    tstep: float = 2. * r / (sqrt(6.) + sqrt(2.))
    t: np.ndarray
    if spacetime in {'black hole', 'Schwarzschild'}:
        raise ValueError('2D black hole spacetime not supported')
    elif spacetime == 'de Sitter':
        t = np.array([-1.5, 0.0, 1.5]) * tstep * (1 + eps)
    else:
        t = np.array([-1.0, 0.0, 1.0]) * tstep * (1 + eps)
    C: List[List[float]] = [[t[0], -s * a, -r_h],
                            [t[1], -s * r_h, -a],
                            [t[0], s * a, -r_h],
                            [t[2], 0., -r],
                            [t[1], s * r, 0.],
                            [t[0], 0., r],
                            [t[1], -s * r_h, a],
                            [t[2], -s * a, r_h],
                            [t[2], s * a, r_h]]
    orientation: str = 'right' if s == 1 else 'left'
    return ('2-simplex rotating \n' + orientation + '-handed (3 steps)', P, C)


def get_2simplexRotated5(edge: float, s: int = 1, eps: float = default_eps,
                         spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 2-simplex rotating right/left-handed (5 steps).
    The rotation is right-handed if s == 1, it is left-handed if s == -1.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    P: List[int] = [7, 10, 3, 1, 12, 5, 2, 14, 8, 11, 4, 15, 13, 6, 9]
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    r_h: float = r / 2  # radius half
    a: float = edge / 2
    tstep: float = 2. * r / (sqrt(6.) + sqrt(2.))
    t: np.ndarray
    if spacetime in {'black hole', 'Schwarzschild'}:
        raise ValueError('2D black hole spacetime not supported')
    elif spacetime == 'de Sitter':
        t = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]) * tstep * (1 + eps)
    else:
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * tstep * (1 + eps)
    C: List[List[float]] = [[t[0], -s * a, -r_h],
                            [t[1], -s * r_h, -a],
                            [t[0], s * a, -r_h],
                            [t[2], 0., -r],
                            [t[1], s * r, 0.],
                            [t[3], s * r_h, -a],
                            [t[0], 0., r],
                            [t[2], s * a, r_h],
                            [t[4], s * a, -r_h],
                            [t[1], -s * r_h, a],
                            [t[3], s * r_h, a],
                            [t[2], -s * a, r_h],
                            [t[4], 0., r],
                            [t[3], -s * r, 0.],
                            [t[4], -s * a, -r_h]]
    orientation: str = 'right' if s == 1 else 'left'
    return ('2-simplex rotating \n' + orientation + '-handed (5 steps)', P, C)


def get_3simplex(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3-simplex.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [6, 4, 12, 2, 10, 1, 8, 7, 14, 5, 13, 3, 11, 9, 15]
    r: float = sqrt(3. / 8.) * edge  # radius (from vertex to center)
    a: float = edge / 2
    b: float = a / sqrt(3.)  # from edge center to face center
    c: float = edge / sqrt(24.)  # from face center to center
    tmax: float = a + b + c
    t: np.ndarray = (np.array([0, a, a + b, a + b + c]) - tmax / 2) * (1 + eps)
    # vertices:
    v1: np.ndarray = np.array([0., -a, -b, -2 * c])
    v2: np.ndarray = np.array([0., a, -b, -2 * c])
    v3: np.ndarray = np.array([0., 0., 2. * b, -2 * c])
    v4: np.ndarray = np.array([0., 0., 0., r - c])
    # coordinates:
    C: np.ndarray = np.array([v1,
                              v2,
                              (v1 + v2) / 2,
                              v3,
                              (v1 + v3) / 2,
                              v4,
                              (v1 + v4) / 2,
                              (v2 + v3) / 2,
                              (v1 + v2 + v3) / 3,
                              (v2 + v4) / 2,
                              (v1 + v2 + v4) / 3,
                              (v3 + v4) / 2,
                              (v1 + v3 + v4) / 3,
                              (v2 + v3 + v4) / 3,
                              (v1 + v2 + v3 + v4) / 4])
    C[:, 0] = [t[0], t[0], t[1], t[0], t[1], t[0], t[1],
               t[1], t[2], t[1], t[2], t[1], t[2], t[2], t[3]]
    return ('3-simplex', P, C.tolist())


def get_latticeD2(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 2D-lattice.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
    s: float = edge / 2
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-3 * t, 0 * s],  # 1
                            [-2 * t, -1 * s],  # 2
                            [-1 * t, -2 * s],  # 3
                            [0 * t, -3 * s],  # 4
                            [-2 * t, 1 * s],  # 5
                            [-1 * t, 0 * s],  # 6
                            [0 * t, -1 * s],  # 7
                            [1 * t, -2 * s],  # 8
                            [-1 * t, 2 * s],  # 9
                            [0 * t, 1 * s],  # 10
                            [1 * t, 0 * s],  # 11
                            [2 * t, -1 * s],  # 12
                            [0 * t, 3 * s],  # 13
                            [1 * t, 2 * s],  # 14
                            [2 * t, 1 * s],  # 15
                            [3 * t, 0 * s]]  # 16
    return ('2D-lattice', P, C)


def get_latticeD3_oct(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice of octahedrons.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [1, 13, 29, 41, 8, 23, 38, 5, 20, 35,
                    17, 33, 2, 14, 30, 42, 11, 27, 26, 9, 24, 39, 6, 21, 36, 19,
                    18, 34, 3, 15, 31, 43, 12, 28, 10, 25, 40, 7, 22, 37,
                    4, 16, 32, 44]
    s: float = edge / 2
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-3 * t, 0 * s, 0 * s],
                            [-2 * t, 0 * s, -1 * s],
                            [-1 * t, 0 * s, -2 * s],
                            [0 * t, 0 * s, -3 * s],
                            [-2 * t, -1 * s, 0 * s],
                            [-1 * t, -1 * s, -1 * s],
                            [0 * t, -1 * s, -2 * s],
                            [-2 * t, 1 * s, 0 * s],
                            [-1 * t, 1 * s, -1 * s],
                            [0 * t, 1 * s, -2 * s],
                            [-1 * t, -2 * s, 0 * s],
                            [0 * t, -2 * s, -1 * s],
                            [-2 * t, 0 * s, 1 * s],
                            [-1 * t, 0 * s, 0 * s],
                            [0 * t, 0 * s, -1 * s],
                            [1 * t, 0 * s, -2 * s],  # end part 1
                            [-1 * t, 2 * s, 0 * s],
                            [0 * t, 2 * s, -1 * s],
                            [0 * t, -3 * s, 0 * s],
                            [-1 * t, -1 * s, 1 * s],
                            [0 * t, -1 * s, 0 * s],
                            [1 * t, -1 * s, -1 * s],
                            [-1 * t, 1 * s, 1 * s],
                            [0 * t, 1 * s, 0 * s],
                            [1 * t, 1 * s, -1 * s],
                            [0 * t, 3 * s, 0 * s],
                            [0 * t, -2 * s, 1 * s],
                            [1 * t, -2 * s, 0 * s],  # end part 2
                            [-1 * t, 0 * s, 2 * s],
                            [0 * t, 0 * s, 1 * s],
                            [1 * t, 0 * s, 0 * s],
                            [2 * t, 0 * s, -1 * s],
                            [0 * t, 2 * s, 1 * s],
                            [1 * t, 2 * s, 0 * s],
                            [0 * t, -1 * s, 2 * s],
                            [1 * t, -1 * s, 1 * s],
                            [2 * t, -1 * s, 0 * s],
                            [0 * t, 1 * s, 2 * s],
                            [1 * t, 1 * s, 1 * s],
                            [2 * t, 1 * s, 0 * s],
                            [0 * t, 0 * s, 3 * s],
                            [1 * t, 0 * s, 2 * s],
                            [2 * t, 0 * s, 1 * s],
                            [3 * t, 0 * s, 0 * s]]  # end part 3
    return ('3D-lattice of octahedrons', P, C)


def get_latticeD4_oct(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 4D-lattice of octahedrons.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = []
    s: float = edge / 2
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-3 * t, 0 * s, 0 * s, 0 * s],
                            [-2 * t, -1 * s, 0 * s, 0 * s],
                            [-2 * t, 0 * s, -1 * s, 0 * s],
                            [-2 * t, 0 * s, 0 * s, -1 * s],
                            [-2 * t, 0 * s, 0 * s, 1 * s],
                            [-2 * t, 0 * s, 1 * s, 0 * s],
                            [-2 * t, 1 * s, 0 * s, 0 * s],
                            [-1 * t, -2 * s, 0 * s, 0 * s],
                            [-1 * t, -1 * s, -1 * s, 0 * s],
                            [-1 * t, -1 * s, 0 * s, -1 * s],
                            [-1 * t, -1 * s, 0 * s, 1 * s],
                            [-1 * t, -1 * s, 1 * s, 0 * s],
                            [-1 * t, 0 * s, -2 * s, 0 * s],
                            [-1 * t, 0 * s, -1 * s, -1 * s],
                            [-1 * t, 0 * s, -1 * s, 1 * s],
                            [-1 * t, 0 * s, 0 * s, -2 * s],
                            [-1 * t, 0 * s, 0 * s, 0 * s],
                            [-1 * t, 0 * s, 0 * s, 2 * s],
                            [-1 * t, 0 * s, 1 * s, -1 * s],
                            [-1 * t, 0 * s, 1 * s, 1 * s],
                            [-1 * t, 0 * s, 2 * s, 0 * s],
                            [-1 * t, 1 * s, -1 * s, 0 * s],
                            [-1 * t, 1 * s, 0 * s, -1 * s],
                            [-1 * t, 1 * s, 0 * s, 1 * s],
                            [-1 * t, 1 * s, 1 * s, 0 * s],
                            [-1 * t, 2 * s, 0 * s, 0 * s],
                            [0 * t, -3 * s, 0 * s, 0 * s],
                            [0 * t, -2 * s, 0 * s, -1 * s],
                            [0 * t, -1 * s, 0 * s, -2 * s],
                            [0 * t, 0 * s, 0 * s, -3 * s],
                            [0 * t, -2 * s, -1 * s, 0 * s],
                            [0 * t, -1 * s, -1 * s, -1 * s],
                            [0 * t, 0 * s, -1 * s, -2 * s],
                            [0 * t, -2 * s, 1 * s, 0 * s],
                            [0 * t, -1 * s, 1 * s, -1 * s],
                            [0 * t, 0 * s, 1 * s, -2 * s],
                            [0 * t, -1 * s, -2 * s, 0 * s],
                            [0 * t, 0 * s, -2 * s, -1 * s],
                            [0 * t, -2 * s, 0 * s, 1 * s],
                            [0 * t, -1 * s, 0 * s, 0 * s],
                            [0 * t, 0 * s, 0 * s, -1 * s],
                            [0 * t, 1 * s, 0 * s, -2 * s],
                            [0 * t, -1 * s, 2 * s, 0 * s],
                            [0 * t, 0 * s, 2 * s, -1 * s],
                            [0 * t, 0 * s, -3 * s, 0 * s],
                            [0 * t, -1 * s, -1 * s, 1 * s],
                            [0 * t, 0 * s, -1 * s, 0 * s],
                            [0 * t, 1 * s, -1 * s, -1 * s],
                            [0 * t, -1 * s, 1 * s, 1 * s],
                            [0 * t, 0 * s, 1 * s, 0 * s],
                            [0 * t, 1 * s, 1 * s, -1 * s],
                            [0 * t, 0 * s, 3 * s, 0 * s],
                            [0 * t, 0 * s, -2 * s, 1 * s],
                            [0 * t, 1 * s, -2 * s, 0 * s],
                            [0 * t, -1 * s, 0 * s, 2 * s],
                            [0 * t, 0 * s, 0 * s, 1 * s],
                            [0 * t, 1 * s, 0 * s, 0 * s],
                            [0 * t, 2 * s, 0 * s, -1 * s],
                            [0 * t, 0 * s, 2 * s, 1 * s],
                            [0 * t, 1 * s, 2 * s, 0 * s],
                            [0 * t, 0 * s, -1 * s, 2 * s],
                            [0 * t, 1 * s, -1 * s, 1 * s],
                            [0 * t, 2 * s, -1 * s, 0 * s],
                            [0 * t, 0 * s, 1 * s, 2 * s],
                            [0 * t, 1 * s, 1 * s, 1 * s],
                            [0 * t, 2 * s, 1 * s, 0 * s],
                            [0 * t, 0 * s, 0 * s, 3 * s],
                            [0 * t, 1 * s, 0 * s, 2 * s],
                            [0 * t, 2 * s, 0 * s, 1 * s],
                            [0 * t, 3 * s, 0 * s, 0 * s],
                            [1 * t, -2 * s, 0 * s, 0 * s],
                            [1 * t, -1 * s, -1 * s, 0 * s],
                            [1 * t, -1 * s, 0 * s, -1 * s],
                            [1 * t, -1 * s, 0 * s, 1 * s],
                            [1 * t, -1 * s, 1 * s, 0 * s],
                            [1 * t, 0 * s, -2 * s, 0 * s],
                            [1 * t, 0 * s, -1 * s, -1 * s],
                            [1 * t, 0 * s, -1 * s, 1 * s],
                            [1 * t, 0 * s, 0 * s, -2 * s],
                            [1 * t, 0 * s, 0 * s, 0 * s],
                            [1 * t, 0 * s, 0 * s, 2 * s],
                            [1 * t, 0 * s, 1 * s, -1 * s],
                            [1 * t, 0 * s, 1 * s, 1 * s],
                            [1 * t, 0 * s, 2 * s, 0 * s],
                            [1 * t, 1 * s, -1 * s, 0 * s],
                            [1 * t, 1 * s, 0 * s, -1 * s],
                            [1 * t, 1 * s, 0 * s, 1 * s],
                            [1 * t, 1 * s, 1 * s, 0 * s],
                            [1 * t, 2 * s, 0 * s, 0 * s],
                            [2 * t, -1 * s, 0 * s, 0 * s],
                            [2 * t, 0 * s, -1 * s, 0 * s],
                            [2 * t, 0 * s, 0 * s, -1 * s],
                            [2 * t, 0 * s, 0 * s, 1 * s],
                            [2 * t, 0 * s, 1 * s, 0 * s],
                            [2 * t, 1 * s, 0 * s, 0 * s],
                            [3 * t, 0 * s, 0 * s, 0 * s]]
    return ('4D-lattice of octahedrons', P, C)


def get_latticeD3_oct_cut(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice of octahedrons (diamond cut).
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [1, 11, 21, 31, 8, 18, 28, 5, 15, 25,
                    2, 12, 22, 32, 9, 19, 29, 6, 16, 26,
                    3, 13, 23, 33, 10, 20, 30, 7, 17, 27,
                    4, 14, 24, 34]
    s: float = edge / 2
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-3 * t, 0 * s, 0 * s],
                            [-2 * t, 0 * s, -1 * s],
                            [-1 * t, 0 * s, -2 * s],
                            [0 * t, 0 * s, -3 * s],
                            [-2 * t, -1 * s, 0 * s],
                            [-1 * t, -1 * s, -1 * s],
                            [0 * t, -1 * s, -2 * s],
                            [-2 * t, 1 * s, 0 * s],
                            [-1 * t, 1 * s, -1 * s],
                            [0 * t, 1 * s, -2 * s],  # end part 1
                            [-2 * t, 0 * s, 1 * s],
                            [-1 * t, 0 * s, 0 * s],
                            [0 * t, 0 * s, -1 * s],
                            [1 * t, 0 * s, -2 * s],
                            [-1 * t, -1 * s, 1 * s],
                            [0 * t, -1 * s, 0 * s],
                            [1 * t, -1 * s, -1 * s],
                            [-1 * t, 1 * s, 1 * s],
                            [0 * t, 1 * s, 0 * s],
                            [1 * t, 1 * s, -1 * s],  # end part 2
                            [-1 * t, 0 * s, 2 * s],
                            [0 * t, 0 * s, 1 * s],
                            [1 * t, 0 * s, 0 * s],
                            [2 * t, 0 * s, -1 * s],
                            [0 * t, -1 * s, 2 * s],
                            [1 * t, -1 * s, 1 * s],
                            [2 * t, -1 * s, 0 * s],
                            [0 * t, 1 * s, 2 * s],
                            [1 * t, 1 * s, 1 * s],
                            [2 * t, 1 * s, 0 * s],  # end part 3
                            [0 * t, 0 * s, 3 * s],
                            [1 * t, 0 * s, 2 * s],
                            [2 * t, 0 * s, 1 * s],
                            [3 * t, 0 * s, 0 * s]]
    return ('3D-lattice of octahedrons', P, move_centre(C))


def get_latticeD3_hcp(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [13, 9, 29, 25, 45, 41, 61, 57, 5, 1, 21, 17, 37, 33, 53, 49,
                    14, 10, 30, 26, 46, 42, 62, 58, 6, 2, 22, 18, 38, 34, 54, 50,
                    15, 11, 31, 27, 47, 43, 63, 59, 7, 3, 23, 19, 39, 35, 55, 51,
                    16, 12, 32, 28, 48, 44, 64, 60, 8, 4, 24, 20, 40, 36, 56, 52]
    a: float = edge / 2
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    t: float = r * (1 + eps)
    C: List[List[float]] = [[-3 * t, 0 * a, -2.5 * r],
                            [-2 * t, -1 * a, -2.0 * r],
                            [-1 * t, -2 * a, -2.5 * r],
                            [0 * t, -3 * a, -2.0 * r],  # 1
                            [-3 * t, 0 * a, 0.5 * r],
                            [-2 * t, -1 * a, 1.0 * r],
                            [-1 * t, -2 * a, 0.5 * r],
                            [0 * t, -3 * a, 1.0 * r],  # 2
                            [-3 * t, 1 * a, -1.0 * r],
                            [-2 * t, 0 * a, -0.5 * r],
                            [-1 * t, -1 * a, -1.0 * r],
                            [0 * t, -2 * a, -0.5 * r],  # 3
                            [-3 * t, 1 * a, 2.0 * r],
                            [-2 * t, 0 * a, 2.5 * r],
                            [-1 * t, -1 * a, 2.0 * r],
                            [0 * t, -2 * a, 2.5 * r],  # 4
                            [-2 * t, 1 * a, -2.0 * r],
                            [-1 * t, 0 * a, -2.5 * r],
                            [0 * t, -1 * a, -2.0 * r],
                            [1 * t, -2 * a, -2.5 * r],  # 1
                            [-2 * t, 1 * a, 1.0 * r],
                            [-1 * t, 0 * a, 0.5 * r],
                            [0 * t, -1 * a, 1.0 * r],
                            [1 * t, -2 * a, 0.5 * r],  # 2
                            [-2 * t, 2 * a, -0.5 * r],
                            [-1 * t, 1 * a, -1.0 * r],
                            [0 * t, 0 * a, -0.5 * r],
                            [1 * t, -1 * a, -1.0 * r],  # 3
                            [-2 * t, 2 * a, 2.5 * r],
                            [-1 * t, 1 * a, 2.0 * r],
                            [0 * t, 0 * a, 2.5 * r],
                            [1 * t, -1 * a, 2.0 * r],  # 4
                            [-1 * t, 2 * a, -2.5 * r],
                            [0 * t, 1 * a, -2.0 * r],
                            [1 * t, 0 * a, -2.5 * r],
                            [2 * t, -1 * a, -2.0 * r],  # 1
                            [-1 * t, 2 * a, 0.5 * r],
                            [0 * t, 1 * a, 1.0 * r],
                            [1 * t, 0 * a, 0.5 * r],
                            [2 * t, -1 * a, 1.0 * r],  # 2
                            [-1 * t, 3 * a, -1.0 * r],
                            [0 * t, 2 * a, -0.5 * r],
                            [1 * t, 1 * a, -1.0 * r],
                            [2 * t, 0 * a, -0.5 * r],  # 3
                            [-1 * t, 3 * a, 2.0 * r],
                            [0 * t, 2 * a, 2.5 * r],
                            [1 * t, 1 * a, 2.0 * r],
                            [2 * t, 0 * a, 2.5 * r],  # 4
                            [0 * t, 3 * a, -2.0 * r],
                            [1 * t, 2 * a, -2.5 * r],
                            [2 * t, 1 * a, -2.0 * r],
                            [3 * t, 0 * a, -2.5 * r],  # 1
                            [0 * t, 3 * a, 1.0 * r],
                            [1 * t, 2 * a, 0.5 * r],
                            [2 * t, 1 * a, 1.0 * r],
                            [3 * t, 0 * a, 0.5 * r],  # 2
                            [0 * t, 4 * a, -0.5 * r],
                            [1 * t, 3 * a, -1.0 * r],
                            [2 * t, 2 * a, -0.5 * r],
                            [3 * t, 1 * a, -1.0 * r],  # 3
                            [0 * t, 4 * a, 2.5 * r],
                            [1 * t, 3 * a, 2.0 * r],
                            [2 * t, 2 * a, 2.5 * r],
                            [3 * t, 1 * a, 2.0 * r]]  # 4
    return ('3D-lattice in HCP', P, move_centre(C))


def get_latticeD3_fcc(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D fcc-lattice.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [1, 9, 21, 37, 5, 17, 33, 49, 13, 29, 45, 57, 25, 41, 53, 61,
                    2, 10, 22, 38, 6, 18, 34, 50, 14, 30, 46, 58, 26, 42, 54, 62,
                    3, 11, 23, 39, 7, 19, 35, 51, 15, 31, 47, 59, 27, 43, 55, 63,
                    4, 12, 24, 40, 8, 20, 36, 52, 16, 32, 48, 60, 28, 44, 56, 64]
    a: float = edge / 2
    r: float = edge / sqrt(3.)  # radius (from vertex to center)
    b: float = r / 2
    t: float = r * (1 + eps)
    C: np.ndarray = np.array([[0 * t, 0 * a, 0 * b],
                              [1 * t, 1 * a, 1 * b],
                              [2 * t, 2 * a, 2 * b],
                              [3 * t, 3 * a, 3 * b],
                              [1 * t, -1 * a, 1 * b],
                              [2 * t, 0 * a, 2 * b],
                              [3 * t, 1 * a, 3 * b],
                              [4 * t, 2 * a, 4 * b],  # 8
                              [1 * t, 0 * a, -1 * b],
                              [2 * t, 1 * a, 0 * b],
                              [3 * t, 2 * a, 1 * b],
                              [4 * t, 3 * a, 2 * b],
                              [2 * t, -2 * a, 2 * b],
                              [3 * t, -1 * a, 3 * b],
                              [4 * t, 0 * a, 4 * b],
                              [5 * t, 1 * a, 5 * b],  # 16
                              [2 * t, -1 * a, 0 * b],
                              [3 * t, 0 * a, 1 * b],
                              [4 * t, 1 * a, 2 * b],
                              [5 * t, 2 * a, 3 * b],
                              [2 * t, 0 * a, -2 * b],
                              [3 * t, 1 * a, -1 * b],
                              [4 * t, 2 * a, 0 * b],
                              [5 * t, 3 * a, 1 * b],  # 24
                              [3 * t, -3 * a, 3 * b],
                              [4 * t, -2 * a, 4 * b],
                              [5 * t, -1 * a, 5 * b],
                              [6 * t, 0 * a, 6 * b],  # end 1
                              [3 * t, -2 * a, 1 * b],
                              [4 * t, -1 * a, 2 * b],
                              [5 * t, 0 * a, 3 * b],
                              [6 * t, 1 * a, 4 * b],  # 32
                              [3 * t, -1 * a, -1 * b],
                              [4 * t, 0 * a, 0 * b],
                              [5 * t, 1 * a, 1 * b],
                              [6 * t, 2 * a, 2 * b],
                              [3 * t, 0 * a, -3 * b],
                              [4 * t, 1 * a, -2 * b],
                              [5 * t, 2 * a, -1 * b],
                              [6 * t, 3 * a, 0 * b],  # 40
                              [4 * t, -3 * a, 2 * b],
                              [5 * t, -2 * a, 3 * b],
                              [6 * t, -1 * a, 4 * b],
                              [7 * t, 0 * a, 5 * b],  # end 2
                              [4 * t, -2 * a, 0 * b],
                              [5 * t, -1 * a, 1 * b],
                              [6 * t, 0 * a, 2 * b],
                              [7 * t, 1 * a, 3 * b],  # 48
                              [4 * t, -1 * a, -2 * b],
                              [5 * t, 0 * a, -1 * b],
                              [6 * t, 1 * a, 0 * b],
                              [7 * t, 2 * a, 1 * b],
                              [5 * t, -3 * a, 1 * b],
                              [6 * t, -2 * a, 2 * b],
                              [7 * t, -1 * a, 3 * b],
                              [8 * t, 0 * a, 4 * b],  # end 3
                              [5 * t, -2 * a, -1 * b],
                              [6 * t, -1 * a, 0 * b],
                              [7 * t, 0 * a, 1 * b],
                              [8 * t, 1 * a, 2 * b],
                              [6 * t, -3 * a, 0 * b],
                              [7 * t, -2 * a, 1 * b],
                              [8 * t, -1 * a, 2 * b],
                              [9 * t, 0 * a, 3 * b]])
    C[:, 0] = C[:, 0] - 4.5 * t
    return ('3D-lattice in FCC', P, move_centre(C.tolist()))


def get_latticeD3_rho(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice of rhombohedrons.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [13, 9, 29, 25, 45, 41, 61, 57, 5, 1, 21, 17, 37, 33, 53, 49,
                    14, 10, 30, 26, 46, 42, 62, 58, 6, 2, 22, 18, 38, 34, 54, 50,
                    15, 11, 31, 27, 47, 43, 63, 59, 7, 3, 23, 19, 39, 35, 55, 51,
                    16, 12, 32, 28, 48, 44, 64, 60, 8, 4, 24, 20, 40, 36, 56, 52]
    s: float = edge / 2
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-3 * t, -1.5 * s, 0 * s],
                            [-2 * t, -1.5 * s, -1 * s],
                            [-1 * t, -1.5 * s, -2 * s],
                            [0 * t, -1.5 * s, -3 * s],
                            [-3 * t, 0.5 * s, 1 * s],
                            [-2 * t, 0.5 * s, 0 * s],
                            [-1 * t, 0.5 * s, -1 * s],
                            [0 * t, 0.5 * s, -2 * s],
                            [-3 * t, -0.5 * s, 0 * s],
                            [-2 * t, -0.5 * s, -1 * s],
                            [-1 * t, -0.5 * s, -2 * s],
                            [0 * t, -0.5 * s, -3 * s],
                            [-3 * t, 1.5 * s, 1 * s],
                            [-2 * t, 1.5 * s, 0 * s],
                            [-1 * t, 1.5 * s, -1 * s],
                            [0 * t, 1.5 * s, -2 * s],
                            [-2 * t, -1.5 * s, 1 * s],
                            [-1 * t, -1.5 * s, 0 * s],
                            [0 * t, -1.5 * s, -1 * s],
                            [1 * t, -1.5 * s, -2 * s],
                            [-2 * t, 0.5 * s, 2 * s],
                            [-1 * t, 0.5 * s, 1 * s],
                            [0 * t, 0.5 * s, 0 * s],
                            [1 * t, 0.5 * s, -1 * s],
                            [-2 * t, -0.5 * s, 1 * s],
                            [-1 * t, -0.5 * s, 0 * s],
                            [0 * t, -0.5 * s, -1 * s],
                            [1 * t, -0.5 * s, -2 * s],
                            [-2 * t, 1.5 * s, 2 * s],
                            [-1 * t, 1.5 * s, 1 * s],
                            [0 * t, 1.5 * s, 0 * s],
                            [1 * t, 1.5 * s, -1 * s],
                            [-1 * t, -1.5 * s, 2 * s],
                            [0 * t, -1.5 * s, 1 * s],
                            [1 * t, -1.5 * s, 0 * s],
                            [2 * t, -1.5 * s, -1 * s],
                            [-1 * t, 0.5 * s, 3 * s],
                            [0 * t, 0.5 * s, 2 * s],
                            [1 * t, 0.5 * s, 1 * s],
                            [2 * t, 0.5 * s, 0 * s],
                            [-1 * t, -0.5 * s, 2 * s],
                            [0 * t, -0.5 * s, 1 * s],
                            [1 * t, -0.5 * s, 0 * s],
                            [2 * t, -0.5 * s, -1 * s],
                            [-1 * t, 1.5 * s, 3 * s],
                            [0 * t, 1.5 * s, 2 * s],
                            [1 * t, 1.5 * s, 1 * s],
                            [2 * t, 1.5 * s, 0 * s],
                            [0 * t, -1.5 * s, 3 * s],
                            [1 * t, -1.5 * s, 2 * s],
                            [2 * t, -1.5 * s, 1 * s],
                            [3 * t, -1.5 * s, 0 * s],
                            [0 * t, 0.5 * s, 4 * s],
                            [1 * t, 0.5 * s, 3 * s],
                            [2 * t, 0.5 * s, 2 * s],
                            [3 * t, 0.5 * s, 1 * s],
                            [0 * t, -0.5 * s, 3 * s],
                            [1 * t, -0.5 * s, 2 * s],
                            [2 * t, -0.5 * s, 1 * s],
                            [3 * t, -0.5 * s, 0 * s],
                            [0 * t, 1.5 * s, 4 * s],
                            [1 * t, 1.5 * s, 3 * s],
                            [2 * t, 1.5 * s, 2 * s],
                            [3 * t, 1.5 * s, 1 * s]]
    return ('3D-lattice of rhombohedrons', P, move_centre(C))


def get_latticeD3_slab(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice slab.
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [17, 13, 27, 9, 22,
                    6, 18, 32, 4, 14, 28, 42, 2, 10, 23, 37,
                    1, 7, 19, 33, 47, 5, 15, 29, 43, 57, 3, 11, 24, 38, 52,
                    8, 20, 34, 48, 62, 16, 30, 44, 58, 71, 12, 25, 39, 53, 67,
                    21, 35, 49, 63, 75, 31, 45, 59, 72, 80, 26, 40, 54, 68, 78,
                    36, 50, 64, 76, 82, 46, 60, 73, 81, 41, 55, 69, 79,
                    51, 65, 77, 61, 74, 56, 70, 66]
    s: float = edge / sqrt(2)
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-5 * t, -2 * s, 0 * s],
                            [-5 * t, -1 * s, -1 * s],
                            [-4 * t, -2 * s, -1 * s],
                            [-5 * t, -1 * s, 1 * s],
                            [-4 * t, -2 * s, 1 * s],
                            [-5 * t, 0 * s, 0 * s],
                            [-4 * t, -1 * s, 0 * s],
                            [-3 * t, -2 * s, 0 * s],
                            [-5 * t, 1 * s, -1 * s],
                            [-4 * t, 0 * s, -1 * s],  # 10
                            [-3 * t, -1 * s, -1 * s],
                            [-2 * t, -2 * s, -1 * s],
                            [-5 * t, 1 * s, 1 * s],
                            [-4 * t, 0 * s, 1 * s],
                            [-3 * t, -1 * s, 1 * s],
                            [-2 * t, -2 * s, 1 * s],
                            [-5 * t, 2 * s, 0 * s],
                            [-4 * t, 1 * s, 0 * s],
                            [-3 * t, 0 * s, 0 * s],
                            [-2 * t, -1 * s, 0 * s],  # 20
                            [-1 * t, -2 * s, 0 * s],
                            [-4 * t, 2 * s, -1 * s],
                            [-3 * t, 1 * s, -1 * s],
                            [-2 * t, 0 * s, -1 * s],
                            [-1 * t, -1 * s, -1 * s],
                            [0 * t, -2 * s, -1 * s],
                            [-4 * t, 2 * s, 1 * s],
                            [-3 * t, 1 * s, 1 * s],
                            [-2 * t, 0 * s, 1 * s],
                            [-1 * t, -1 * s, 1 * s],  # 30
                            [0 * t, -2 * s, 1 * s],
                            [-3 * t, 2 * s, 0 * s],
                            [-2 * t, 1 * s, 0 * s],
                            [-1 * t, 0 * s, 0 * s],
                            [0 * t, -1 * s, 0 * s],
                            [1 * t, -2 * s, 0 * s],
                            [-2 * t, 2 * s, -1 * s],
                            [-1 * t, 1 * s, -1 * s],
                            [0 * t, 0 * s, -1 * s],
                            [1 * t, -1 * s, -1 * s],  # 40
                            [2 * t, -2 * s, -1 * s],
                            [-2 * t, 2 * s, 1 * s],
                            [-1 * t, 1 * s, 1 * s],
                            [0 * t, 0 * s, 1 * s],
                            [1 * t, -1 * s, 1 * s],
                            [2 * t, -2 * s, 1 * s],
                            [-1 * t, 2 * s, 0 * s],
                            [0 * t, 1 * s, 0 * s],
                            [1 * t, 0 * s, 0 * s],
                            [2 * t, -1 * s, 0 * s],  # 50
                            [3 * t, -2 * s, 0 * s],
                            [0 * t, 2 * s, -1 * s],
                            [1 * t, 1 * s, -1 * s],
                            [2 * t, 0 * s, -1 * s],
                            [3 * t, -1 * s, -1 * s],
                            [4 * t, -2 * s, -1 * s],
                            [0 * t, 2 * s, 1 * s],
                            [1 * t, 1 * s, 1 * s],
                            [2 * t, 0 * s, 1 * s],
                            [3 * t, -1 * s, 1 * s],  # 60
                            [4 * t, -2 * s, 1 * s],
                            [1 * t, 2 * s, 0 * s],
                            [2 * t, 1 * s, 0 * s],
                            [3 * t, 0 * s, 0 * s],
                            [4 * t, -1 * s, 0 * s],
                            [5 * t, -2 * s, 0 * s],
                            [2 * t, 2 * s, -1 * s],
                            [3 * t, 1 * s, -1 * s],
                            [4 * t, 0 * s, -1 * s],
                            [5 * t, -1 * s, -1 * s],  # 70
                            [2 * t, 2 * s, 1 * s],
                            [3 * t, 1 * s, 1 * s],
                            [4 * t, 0 * s, 1 * s],
                            [5 * t, -1 * s, 1 * s],
                            [3 * t, 2 * s, 0 * s],
                            [4 * t, 1 * s, 0 * s],
                            [5 * t, 0 * s, 0 * s],
                            [4 * t, 2 * s, -1 * s],
                            [5 * t, 1 * s, -1 * s],
                            [4 * t, 2 * s, 1 * s],  # 80
                            [5 * t, 1 * s, 1 * s],
                            [5 * t, 2 * s, 0 * s]]
    return ('3D-lattice (slab)', P, move_centre(C))


def get_latticeD3_slabpinf(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice slab past (4-step past infinity). 
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [17, 13, 24, 9, 21,
                    6, 18, 27, 4, 14, 25, 30, 2, 10, 22, 29,
                    1, 7, 19, 28, 5, 15, 26, 3, 11, 23,
                    8, 20, 16, 12]
    s: float = edge / sqrt(2)
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-5 * t, -2 * s, 0 * s],
                            [-5 * t, -1 * s, -1 * s],
                            [-4 * t, -2 * s, -1 * s],
                            [-5 * t, -1 * s, 1 * s],
                            [-4 * t, -2 * s, 1 * s],
                            [-5 * t, 0 * s, 0 * s],
                            [-4 * t, -1 * s, 0 * s],
                            [-3 * t, -2 * s, 0 * s],
                            [-5 * t, 1 * s, -1 * s],
                            [-4 * t, 0 * s, -1 * s],  # 10
                            [-3 * t, -1 * s, -1 * s],
                            [-2 * t, -2 * s, -1 * s],
                            [-5 * t, 1 * s, 1 * s],
                            [-4 * t, 0 * s, 1 * s],
                            [-3 * t, -1 * s, 1 * s],
                            [-2 * t, -2 * s, 1 * s],
                            [-5 * t, 2 * s, 0 * s],
                            [-4 * t, 1 * s, 0 * s],
                            [-3 * t, 0 * s, 0 * s],
                            [-2 * t, -1 * s, 0 * s],  # 20
                            [-4 * t, 2 * s, -1 * s],
                            [-3 * t, 1 * s, -1 * s],
                            [-2 * t, 0 * s, -1 * s],
                            [-4 * t, 2 * s, 1 * s],
                            [-3 * t, 1 * s, 1 * s],
                            [-2 * t, 0 * s, 1 * s],
                            [-3 * t, 2 * s, 0 * s],
                            [-2 * t, 1 * s, 0 * s],
                            [-2 * t, 2 * s, -1 * s],
                            [-2 * t, 2 * s, 1 * s]]  # 30
    return ('3D-lattice (slab, past inf.)', P, move_centre(C))


def get_latticeD3_slabfinf(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a 3D-lattice slab future (4-step future infinity). 
    The first argument set the size, the optional second argument sets 
    a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [19, 15, 11, 23,
                    8, 20, 28, 5, 16, 26, 3, 12, 24, 30,
                    2, 9, 21, 29, 1, 6, 17, 27, 4, 13, 25,
                    10, 22, 7, 18, 14]
    s: float = edge / sqrt(2)
    t: float = s * (1 + eps)
    C: List[List[float]] = [[2 * t, -2 * s, -1 * s],
                            [2 * t, -2 * s, 1 * s],
                            [2 * t, -1 * s, 0 * s],
                            [3 * t, -2 * s, 0 * s],
                            [2 * t, 0 * s, -1 * s],
                            [3 * t, -1 * s, -1 * s],
                            [4 * t, -2 * s, -1 * s],
                            [2 * t, 0 * s, 1 * s],
                            [3 * t, -1 * s, 1 * s],
                            [4 * t, -2 * s, 1 * s],  # 10
                            [2 * t, 1 * s, 0 * s],
                            [3 * t, 0 * s, 0 * s],
                            [4 * t, -1 * s, 0 * s],
                            [5 * t, -2 * s, 0 * s],
                            [2 * t, 2 * s, -1 * s],
                            [3 * t, 1 * s, -1 * s],
                            [4 * t, 0 * s, -1 * s],
                            [5 * t, -1 * s, -1 * s],
                            [2 * t, 2 * s, 1 * s],
                            [3 * t, 1 * s, 1 * s],  # 20
                            [4 * t, 0 * s, 1 * s],
                            [5 * t, -1 * s, 1 * s],
                            [3 * t, 2 * s, 0 * s],
                            [4 * t, 1 * s, 0 * s],
                            [5 * t, 0 * s, 0 * s],
                            [4 * t, 2 * s, -1 * s],
                            [5 * t, 1 * s, -1 * s],
                            [4 * t, 2 * s, 1 * s],
                            [5 * t, 1 * s, 1 * s],
                            [5 * t, 2 * s, 0 * s]]  # 30
    return ('3D-lattice (slab, future inf.)', P, move_centre(C))


def get_latticeD3_slabpert(edge: float, eps: float = default_eps, spacetime: str = '') -> \
        Tuple[str, List[int], List[List[float]]]:
    '''
    Returns the name, event permutation (for Hasse diagrams), and event 
    coordinates of a a 3D-lattice slab, perturbed. 
    The first argument set the size, the optional second argument 
    sets a small time offset for each layer.
    '''
    if spacetime != 'Minkowski' and spacetime != '':
        raise ValueError('Spacetime not supported')
    P: List[int] = [17,         13, 27,                9, 22,
                    6, 18, 32,    4, 14, 28, 42,          2, 10, 23, 37,
                    1, 7, 19, 33, 49, 5, 15, 29, 43,      60, 3, 11, 24, 38,      55,
                    8, 20, 34, 51, 65, 16, 30, 35, 46, 45, 61, 74, 12, 25, 39, 50, 56, 70,
                    21, 44, 52, 66, 78, 31, 47,      62, 75, 83, 26, 40,      57, 71, 81,
                    36, 53, 67, 79, 85, 48,      63, 76, 84,   41,      58, 72, 82,
                    54, 68, 80,            64, 77,               59, 73,
                    69]
    s: float = edge / sqrt(2)
    t: float = s * (1 + eps)
    C: List[List[float]] = [[-5 * t, -2 * s, 0 * s],
                            [-5 * t, -1 * s, -1 * s],
                            [-4 * t, -2 * s, -1 * s],
                            [-5 * t, -1 * s, 1 * s],
                            [-4 * t, -2 * s, 1 * s],
                            [-5 * t, 0 * s, 0 * s],
                            [-4 * t, -1 * s, 0 * s],
                            [-3 * t, -2 * s, 0 * s],
                            [-5 * t, 1 * s, -1 * s],
                            [-4 * t, 0 * s, -1 * s],  # 10
                            [-3 * t, -1 * s, -1 * s],
                            [-2 * t, -2 * s, -1 * s],
                            [-5 * t, 1 * s, 1 * s],
                            [-4 * t, 0 * s, 1 * s],
                            [-3 * t, -1 * s, 1 * s],
                            [-2 * t, -2 * s, 1 * s],
                            [-5 * t, 2 * s, 0 * s],
                            [-4 * t, 1 * s, 0 * s],
                            [-3 * t, 0 * s, 0 * s],
                            [-2 * t, -1 * s, 0 * s],  # 20
                            [-1 * t, -2 * s, 0 * s],
                            [-4 * t, 2 * s, -1 * s],
                            [-3 * t, 1 * s, -1 * s],
                            [-2 * t, 0 * s, -1 * s],
                            [-1 * t, -1 * s, -1 * s],  # 25
                            [0 * t, -2 * s, -1 * s],
                            [-4 * t, 2 * s, 1 * s],
                            [-3 * t, 1 * s, 1 * s],
                            [-2 * t, 0 * s, 1 * s],
                            [-1 * t, -1 * s, 1 * s],  # 30
                            [0 * t, -2 * s, 1 * s],
                            [-3 * t, 2 * s, 0 * s],
                            [-2 * t, 1 * s, 0 * s],
                            [-1 * t, 0 * s, 0 * s],
                            [-0.5 * t, -0.5 * s, 0 * s],  # pert 35
                            [1 * t, -2 * s, 0 * s],
                            [-2 * t, 2 * s, -1 * s],
                            [-1 * t, 1 * s, -1 * s],
                            [-0 * t, 0 * s, -1 * s],
                            [1 * t, -1 * s, -1 * s],  # 40
                            [2 * t, -2 * s, -1 * s],
                            [-2 * t, 2 * s, 1 * s],
                            [-1 * t, 1 * s, 1 * s],
                            [0.5 * t, -5 / 6 * s, 5 / 6 * s],  # pert 44
                            [0 * t, 0 * s, 1 * s],
                            [0 * t, 0.5 * s, -0.25 * s],  # pert 46
                            [1 * t, -1 * s, 1 * s],
                            [2 * t, -2 * s, 1 * s],
                            [-1 * t, 2 * s, 0 * s],
                            [0.5 * t, 0.25 * s, 0.15 * s],  # pert 50
                            [0 * t, 1 * s, 0 * s],
                            [1 * t, 0 * s, 0 * s],
                            [2 * t, -1 * s, 0 * s],
                            [3 * t, -2 * s, 0 * s],
                            [0 * t, 2 * s, -1 * s],
                            [1 * t, 1 * s, -1 * s],
                            [2 * t, 0 * s, -1 * s],
                            [3 * t, -1 * s, -1 * s],
                            [4 * t, -2 * s, -1 * s],
                            [0 * t, 2 * s, 1 * s],  # 60
                            [1 * t, 1 * s, 1 * s],
                            [2 * t, 0 * s, 1 * s],
                            [3 * t, -1 * s, 1 * s],
                            [4 * t, -2 * s, 1 * s],
                            [1 * t, 2 * s, 0 * s],
                            [2 * t, 1 * s, 0 * s],
                            [3 * t, 0 * s, 0 * s],
                            [4 * t, -1 * s, 0 * s],
                            [5 * t, -2 * s, 0 * s],
                            [2 * t, 2 * s, -1 * s],  # 70
                            [3 * t, 1 * s, -1 * s],
                            [4 * t, 0 * s, -1 * s],
                            [5 * t, -1 * s, -1 * s],
                            [2 * t, 2 * s, 1 * s],
                            [3 * t, 1 * s, 1 * s],
                            [4 * t, 0 * s, 1 * s],
                            [5 * t, -1 * s, 1 * s],
                            [3 * t, 2 * s, 0 * s],
                            [4 * t, 1 * s, 0 * s],
                            [5 * t, 0 * s, 0 * s],  # 80
                            [4 * t, 2 * s, -1 * s],
                            [5 * t, 1 * s, -1 * s],
                            [4 * t, 2 * s, 1 * s],
                            [5 * t, 1 * s, 1 * s],
                            [5 * t, 2 * s, 0 * s]]
    return ('3D-lattice (slab, perturbed)', P, move_centre(C))
