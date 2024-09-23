from four_circle.utilities.calculations import FourCircle

fc = FourCircle('./data/MnWO4_lattice.dat')

peaks = fc.read_observations('./data/MnWO4_observe.dat')

fc.calculate_UB_from_two_vectors(peaks)

print(fc.UB_matrix())

fc.optimize_lattice(peaks, cell='Monoclinic')

print(fc.UB_matrix())

print(fc.get_lattice_parameters())

fc.azimuthal_scan(0, 0, 1)