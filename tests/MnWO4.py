import numpy as np

from four_circle.utilities.calculations import FourCircle

two_theta_lim = [-2, 92]
chi_lim = [-50, 50]

d_spacing_mask = [2.338, 2.025] # aluminum 111, 200
two_theta_window_mask = 3

wl = 2.38
lat_params = [4.83, 5.76, 4.99, 90, 91.14, 90]

max_strong = 10

peaks_1 = [1, 2, 1, -65.2, -32.75, 24.8, -52.6]
peaks_2 = [1, 2, -1, -64.7, -32.35, -26.5, -41]

peaks = np.column_stack([peaks_1, peaks_2])

cif_file = '/HFIR/HB1A/IPTS-32750/shared/matlab_scripts/MnWO4.cif'

# --- initialize sample/instrument ---
fc = FourCircle()
fc.set_lattice_parameters(lat_params)
fc.set_wavelength(wl)
fc.set_angle_limits(two_theta_lim, chi_lim)

# --- initialize UB ---
fc.calculate_UB_from_two_vectors(peaks)

# --- generate strong nuclear peak scan ---
nuclear_table = fc.generate_reflection_table(cif_file,
                                             max_reflections=max_strong,
                                             d_min=0.7)

fc.g

# fc.optimize_lattice(peaks, cell='Monoclinic')

# print(fc.UB_matrix())

# print(fc.get_lattice_parameters())

# peaks = fc.read_observations('./data/MnWO4_index.dat')

# fc.index_peaks(peaks)

