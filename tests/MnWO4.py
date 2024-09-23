from four_circle.utilities.lattice import Lattice

lat = Lattice('./data/MnWO4_lattice.dat')

peaks = lat.read_observations('./data/MnWO4_observe.dat')

lat.calculate_UB_from_two_vectors(peaks)

print(lat.UB_matrix())

lat.optimize_lattice(peaks, cell='Monoclinic')

print(lat.UB_matrix())

print(lat.get_lattice_parameters())