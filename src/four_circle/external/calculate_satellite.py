# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

from mantid.geometry import CrystalStructure, ReflectionGenerator, ReflectionConditionFilter
from mantid.kernel import V3D

def angular_distance(setting1, setting2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(setting1, setting2)))

def greedy_sort(settings):
    n = len(settings)
    visited = np.zeros(n, dtype=bool)
    path = []
    current_index = 0
    path.append(current_index)
    visited[current_index] = True

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = angular_distance(settings[i], settings[j])

    for _ in range(n - 1):
        next_index = np.argmin([distance_matrix[current_index, j] if not visited[j] else np.inf for j in range(n)])
        visited[next_index] = True
        path.append(next_index)
        current_index = next_index
    return [i for i in path]

CreateSingleValuedWorkspace(OutputWorkspace='sample')

LoadCIF(Workspace='sample', InputFile='/HFIR/HB1A/IPTS-33477/shared/hexaferrite.cif')

generator = ReflectionGenerator(mtd['sample'].sample().getCrystalStructure())

hkls = generator.getHKLsUsingFilter(1.5, 45, ReflectionConditionFilter.StructureFactor)

#hkls = [[0,0,9.86], [0,0,8.14], [0,0,17.13], [0,0,18.86], [0,0,23.15], [0,0,24.86], [0,0,36.86], [0,0,35.14], [1,0,10.87], [1,0,9.13], [0,0,15.86], [0,0,17.15]]

#k1 = V3D(0, 0, 1.5)
#hkls = hkls+[V3D(0,0,0)]

k1 = V3D(0, 0, 0.86)

hkls = [V3D(*(hkl+k1)) for hkl in hkls]+[V3D(*(hkl-k1)) for hkl in hkls]

F2s = np.array(generator.getFsSquared(hkls))

ds = np.array(generator.getDValues(hkls))

hkls = np.array(hkls)

UB = np.loadtxt('/HFIR/HB1A/IPTS-33477/shared/hexaferrite/UBmatrix.dat.hexaferrite')

*lattice, lamda = np.loadtxt('/HFIR/HB1A/IPTS-33477/shared/hexaferrite/lattice.dat')

SetUB(Workspace='sample', UB=UB)

B = mtd['sample'].sample().getOrientedLattice().getB()

Qc = np.einsum('ij,kj->ik', B, hkls)

q = np.sqrt(np.sum(Qc**2, axis=0))

theta = np.arcsin(lamda*q/2)
theta2 = theta*2*180/np.pi

omega = theta2/2

hphi = np.einsum('ij,kj->ik', UB, hkls)

newphi = np.zeros_like(ds)
newchi = np.zeros_like(ds)

for i in range(len(ds)):
    newphi[i] = np.arctan(hphi[1,i]/hphi[0,i])*180/np.pi
    if newphi[i] > 0 and hphi[0,i] < 0:
        newphi[i] -= 180
    elif newphi[i] <= 0 and hphi[0,i] < 0:
        newphi[i] += 180

    newchi[i] = np.arctan(hphi[2,i]/np.sqrt(hphi[0,i]**2+hphi[1,i]**2))*180/np.pi

mask = (theta2 >= -2) & (theta2 < 80) & (newchi > -70) & (newchi < 40) & (np.abs(theta2-61) > 3) & (np.abs(theta2-72) > 3)

hkls = hkls[mask]
ds = ds[mask]
F2s = F2s[mask]
theta2 = theta2[mask]
omega = omega[mask]
newphi = newphi[mask]
newchi = newchi[mask]

sort = greedy_sort(np.column_stack([theta2, newchi, newphi]))

with open('/HFIR/HB1A/IPTS-33477/shared/hexaferrite/scanlist_Xtal_ICM.dat', 'w') as f:
    for i in sort:
        f.write('{:8.2f}{:8.2f}{:8.2f}{:12.2f}{:12.4f}{:8.2f}{:8.2f}{:8.2f}{:8.2f}\n'.format(*hkls[i], F2s[i], ds[i], theta2[i], omega[i], newchi[i], newphi[i]))
