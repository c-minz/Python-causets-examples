#!/usr/bin/env python
'''
Created on 28 Apr 2021

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List
import unittest
import lightcone_animations_coordinates as LAC
import numpy as np
from causets.causet import Causet
from causets.embeddedcauset import EmbeddedCauset
import causets.causetplotting as cplt
from matplotlib import pyplot as plt


class TestCauset(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_2lattice(self):
        name: str
        perm: List[int]
        coords: List[List[float]]
        name, perm, coords = LAC.get_latticeD3_slabfinf(0.3)

        C: EmbeddedCauset = EmbeddedCauset(coordinates=coords)
#         cplt.plot(C, dims=[1, 2, 0], labels=True,
#                   axislim={'xlim': [-1.0, 1.0], 'ylim': [-1.0, 1.0],
#                            'zlim': [-1.0, 1.0]})
        cplt.plotDiagram([C.find(i) for i in perm], perm,
                         labels=True, links={'linewidth': 1.5, 'alpha': 0.5},
                         axislim={'xlim': [-1.0, 1.0], 'ylim': [-1.0, 1.0]})
        plt.show()

        # print links:
        L: np.ndarray = C.LinkFutureMatrix(C.sortedByLabels())
        n: int = L.shape[1]
        s: str = ''.join([f'{i + 1}/{j + 1},' for i in range(n)
                          for j in range(n) if L[i, j]])
        print(s)


if __name__ == '__main__':
    unittest.main()
