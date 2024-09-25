from four_circle.utilities.calculations import FourCircle

two_theta_lim = [-2, 92]
chi_lim = [-50, 50]

d_spacing_mask = [2.338, 2.025] # aluminum 111, 200
two_theta_window_mask = 3

fc = FourCircle('./data/MnWO4_lattice.dat')

peaks = fc.read_observations('./data/MnWO4_observe.dat')

fc.calculate_UB_from_two_vectors(peaks)

print(fc.UB_matrix())

fc.optimize_lattice(peaks, cell='Monoclinic')

print(fc.UB_matrix())

print(fc.get_lattice_parameters())

peaks = fc.read_observations('./data/MnWO4_index.dat')

fc.index_peaks(peaks)

