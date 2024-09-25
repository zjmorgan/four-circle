import numpy as np

from mantid.simpleapi import SetUB, LoadCIF, CreateSingleValuedWorkspace, mtd
from mantid.geometry import ReflectionGenerator, ReflectionConditionFilter

scan_list = '/HFIR/HB1A/IPTS-32750/shared/matlab_scripts/MnWO4_MATLAB/scanlist_Xtal.dat'
cif_file = '/HFIR/HB1A/IPTS-32750/shared/matlab_scripts/MnWO4.cif'
ub_file = '/HFIR/HB1A/IPTS-32750/shared/matlab_scripts/MnWO4_MATLAB/UBmatrix.dat'

lamda = 2.38

d_min = 1.5
d_max = 4.5

two_theta_min = -2
two_theta_max = 87

chi_min = -50
chi_max = 50

two_theta_tol = 3
two_theta_al1 = 61
two_theta_al2 = 72

def angular_distance(setting_1, setting_2):

    return np.sqrt(sum((a-b)**2 for a, b in zip(setting_1, setting_2)))

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
            distance_matrix[i,j] = angular_distance(settings[i], settings[j])

    for _ in range(n - 1):
        next_index = np.argmin([distance_matrix[current_index,j] if not visited[j] else np.inf for j in range(n)])
        visited[next_index] = True
        path.append(next_index)
        current_index = next_index
    return [i for i in path]

CreateSingleValuedWorkspace(OutputWorkspace='sample')

LoadCIF(Workspace='sample', InputFile=cif_file)

generator = ReflectionGenerator(mtd['sample'].sample().getCrystalStructure())

hkls = generator.getHKLsUsingFilter(d_min, d_max, ReflectionConditionFilter.StructureFactor)

ds = np.array(generator.getDValues(hkls))
F2s = np.array(generator.getFsSquared(hkls))

hkls = np.array(hkls)

UB = np.loadtxt(ub_file)

SetUB(Workspace='sample', UB=UB)

B = mtd['sample'].sample().getOrientedLattice().getB()

s = np.einsum('ij,kj->ik', B, hkls)

q = np.sqrt(np.sum(s**2, axis=0))

theta = np.arcsin(lamda*q/2)
two_theta = 2*theta

omega = two_theta/2

s = np.einsum('ij,kj->ik', UB, hkls)

newphi = np.zeros_like(ds)
newchi = np.zeros_like(ds)

for i in range(len(ds)):
    newphi[i] = np.arctan(s[1,i]/s[0,i])
    if newphi[i] > 0 and s[0,i] < 0:
        newphi[i] -= np.pi
    elif newphi[i] <= 0 and s[0,i] < 0:
        newphi[i] += np.pi

    newchi[i] = np.arctan2(s[2,i], np.sqrt(s[0,i]**2+s[1,i]**2))

mask = (two_theta >= two_theta_min) & (two_theta < two_theta_max) \
     & (newchi > chi_min) & (newchi < chi_max) \
     & (np.abs(two_theta-two_theta_al1) > two_theta_tol) \
     & (np.abs(two_theta-two_theta_al2) > two_theta_tol)

hkls = hkls[mask]
ds = ds[mask]
F2s = F2s[mask]
two_theta = two_theta[mask]
omega = omega[mask]
newphi = newphi[mask]
newchi = newchi[mask]

sort = greedy_sort(np.column_stack([two_theta, 1.5*newchi, 2*newphi]))

hkl_fmt = '{:4.0f}{:4.0f}{:4.0f}{:12.2f}{:12.4f}{:8.2f}{:8.2f}{:8.2f}{:8.2f}\n'
with open(scan_list, 'w') as f:
    for i in sort:
        line = F2s[i], ds[i], two_theta[i], omega[i], newchi[i], newphi[i]
        f.write(hkl_fmt.format(*hkls[i], *line))
