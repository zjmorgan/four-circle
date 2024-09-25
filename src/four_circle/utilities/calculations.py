import os
import numpy as np

import scipy.optimize
import scipy.linalg
import scipy.spatial

import matplotlib.pyplot as plt

from matid.simpleapi import CreateSingleValuedWorkspace, LoadCIF, mtd
from mantid.geometry import ReflectionGenerator, ReflectionConditionFilter
from mantid.kernel import V3D

class FourCircle:

    def __init__(self, filename=None):

        self.a = self.b = self.c = self.alpha = self.beta = self.gamma = None

        self.lamda = None

        self.theta = self.phi = self.omega = None

        if filename is not None:

            self.read_parameters(filename)

        self.instrument = 'HB1A'

    def get_lattice_parameters(self):

        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def get_wavelength(self):

        return self.lamda

    def set_wavelength(self, lamda):

        self.lamda = lamda

    def get_axis_angles(self):

        return self.u_phi, self.u_theta, self.u_omega

    def set_axis_angles(self, values):

        self.u_phi, self.u_theta, self.u_omega = values

    def set_lattice_parameters(self, params):

        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = params

    def set_angle_limits(self, two_theta_lim=[-2, 90],
                               omega_lim=[-180,180],
                               chi_lim=[-40,40],
                               phi_lim=[-180, 180]):

        self.two_theta_lim = two_theta_lim
        self.chi_lim = chi_lim
        self.omega_lim = omega_lim
        self.phi_lim = phi_lim

    def powder_rings(self, d=[2.338, 2.025], angle=3):

        lamda = self.get_wavelength()

        self.two_theta_prune = np.rad2deg(np.arcsin(lamda/2/d))

        self.two_theta_tol_angle = angle

    def prune_unreachable_settings(self, two_theta, omega, chi, phi):

        two_theta_min, two_theta_max = self.two_theta_lim
        omega_min, omega_max = self.omega_lim
        chi_min, chi_max = self.chi_lim
        phi_min, phi_max = self.phi_lim

        mask = (two_theta > two_theta_min) \
             & (two_theta < two_theta_max) \
             & (omega > omega_min) \
             & (omega < omega_max) \
             & (chi > chi_min) \
             & (chi < chi_max) \
             & (phi > phi_min) \
             & (phi < phi_max)

        return mask

    def read_parameters(self, filename):

        params = []
        with open(filename, 'r') as f:
            for line in f:
                values = line.split()
                for value in values:
                    params.append(float(value))
                    if len(params) == 6:
                        self.set_lattice_parameters(params)
                    elif len(params) == 7:
                        self.set_wavelength(float(value))

    def write_parameters(self, filename):

        params = self.get_lattice_parameters()

        with open(filename, 'w') as f:
            line = ' '.join(7*['{:.4f}']).format(*params)
            f.write(line)

    def read_observations(self, filename):

        obs = []
        with open(filename, 'r') as f:
            for line in f:
                values = np.array(line.split()).astype(float)
                obs.append(values)

        return np.array(obs).T

    def convert_angles(self, peaks):

        h, k, l, two_theta, omega, chi, phi = peaks

        omega -= two_theta/2

        two_theta, omega, chi, phi = np.deg2rad([two_theta, omega, chi, phi])

        return h, k, l, two_theta, omega, chi, phi

    def calculate_U_phi(self, omega, chi, phi):

        U_phi = np.row_stack([np.cos(omega)*np.cos(chi)*np.cos(phi)-np.sin(omega)*np.sin(phi),
                              np.cos(omega)*np.cos(chi)*np.sin(phi)+np.sin(omega)*np.cos(phi),
                              np.cos(omega)*np.sin(chi)])

        return U_phi

    def calculate_scattering_vector(self, two_theta, omega, chi, phi, lamda):

        U_phi = self.calculate_U_phi(omega, chi, phi)

        s = 2*np.sin(two_theta/2)/lamda*U_phi

        return s

    def calculate_UB_from_two_vectors(self, peaks):

        B = self.B_matrix(*self.get_lattice_parameters())

        h, k, l, two_theta, omega, chi, phi = self.convert_angles(peaks)

        hkl_1 = h[0], k[0], l[0]
        hkl_2 = h[1], k[1], l[1]

        h1c = np.dot(B, hkl_1)
        h2c = np.dot(B, hkl_2)
        h3c = np.cross(h1c, h2c)

        t1c = h1c/np.linalg.norm(h1c)
        t3c = h3c/np.linalg.norm(h3c)
        t2c = np.cross(t3c, t1c)
        Tc = np.column_stack([t1c, t2c, t3c])

        U_phi = self.calculate_U_phi(omega, chi, phi)

        h1p = U_phi[:,0]
        h2p = U_phi[:,1]
        ang = np.rad2deg(np.arccos(np.dot(h1p, h2p)/(np.linalg.norm(h1p)*np.linalg.norm(h2p))))

        h3p = np.cross(h1p, h2p)
        t1p = h1p/np.linalg.norm(h1p)
        t3p = h3p/np.linalg.norm(h3p)
        t2p = np.cross(t3p, t1p)
        Tp = np.column_stack((t1p, t2p, t3p))

        U = Tp @ Tc.T

        self.set_orientation_angles(U)

        UB = self.UB_matrix()
        UB_inv = np.linalg.inv(UB)

        lamda = self.get_wavelength()

        print('Calculated (h,k,l) from observation:')
        s = self.calculate_scattering_vector(two_theta, omega, chi, phi, lamda)
        for i in range(two_theta.size):
            h, k, l = UB_inv @ s[:,i]
            print('{:5.3f} {:5.3f} {:5.3f}'.format(h,k,l))

        print('\nAngle between the first two reflections is {}'.format(ang))

    def set_orientation_angles(self, U):

        omega = np.arccos((np.trace(U)-1)/2)

        val, vec = np.linalg.eig(U)

        ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

        theta = np.arccos(uz)
        phi = np.arctan2(uy, ux)

        self.set_axis_angles([phi, theta, omega])

    def U_matrix(self, phi, theta, omega):

        u0 = np.cos(phi)*np.sin(theta)
        u1 = np.sin(phi)*np.sin(theta)
        u2 = np.cos(theta)

        w = omega*np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def B_matrix(self, a, b, c, alpha, beta, gamma):

        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

        G = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                      [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                      [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def UB_matrix(self):

        B = self.B_matrix(*self.get_lattice_parameters())
        U = self.U_matrix(*self.get_axis_angles())

        return np.dot(U, B)

    def cubic(self, x):

        a, *params = x

        return (a, a, a, 90, 90, 90, *params)

    def rhombohedral(self, x):

        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):

        a, c, *params = x

        return (a, a, c, 90, 90, 90, *params)

    def hexagonal(self, x):

        a, c, *params = x

        return (a, a, c, 90, 90, 120, *params)

    def orthorhombic(self, x):

        a, b, c, *params = x

        return (a, b, c, 90, 90, 90, *params)

    def monoclinic(self, x):

        a, b, c, beta, *params = x

        return (a, b, c, 90, beta, 90, *params)

    def triclinic(self, x):

        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def residual(self, x, hkl, s, fun):

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U,B)

        return (np.einsum('ij,jl->il', UB, hkl)-s).flatten()

    def optimize_lattice(self, peaks, cell='Triclinic'):

        h, k, l, two_theta, omega, chi, phi = self.convert_angles(peaks)

        hkl = np.row_stack([h, k, l])

        lamda = self.get_wavelength()

        s = self.calculate_scattering_vector(two_theta, omega, chi, phi, lamda)

        a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

        phi, theta, omega = self.get_axis_angles()

        fun_dict = {'Cubic': self.cubic,
                    'Rhombohedral': self.rhombohedral,
                    'Tetragonal': self.tetragonal,
                    'Hexagonal': self.hexagonal,
                    'Orthorhombic': self.orthorhombic,
                    'Monoclinic': self.monoclinic,
                    'Triclinic': self.triclinic}

        x0_dict = {'Cubic': (a, ),
                   'Rhombohedral': (a, alpha),
                   'Tetragonal': (a, c),
                   'Hexagonal': (a, c),
                   'Orthorhombic': (a, b, c),
                   'Monoclinic': (a, b, c, beta),
                   'Triclinic': (a, b, c, alpha, beta, gamma)}

        fun = fun_dict[cell]
        x0 = x0_dict[cell]

        x0 += (phi, theta, omega)
        args = (hkl, s, fun)

        sol = scipy.optimize.least_squares(self.residual,
                                           x0=x0,
                                           args=args)

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

        J = sol.jac
        cov = np.linalg.inv(J.T.dot(J))

        chi2dof = np.sum(sol.fun**2)/(sol.fun.size-sol.x.size)
        cov *= chi2dof

        sig = np.sqrt(np.diagonal(cov))

        sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

        if np.isclose(a, sig_a):
            sig_a = 0
        if np.isclose(b, sig_b):
            sig_b = 0
        if np.isclose(c, sig_c):
            sig_c = 0

        if np.isclose(alpha, sig_alpha):
            sig_alpha = 0
        if np.isclose(beta, sig_beta):
            sig_beta = 0
        if np.isclose(gamma, sig_gamma):
            sig_gamma = 0

        self.set_lattice_parameters([a, b, c, alpha, beta, gamma])
        self.set_axis_angles([phi, theta, omega])

    def index_peaks(self, peaks):

        h, k, l, two_theta, omega, chi, phi = self.convert_angles(peaks)

        hkl = np.row_stack([h, k, l])

        lamda = self.get_wavelength()

        s = self.calculate_scattering_vector(two_theta, omega, chi, phi, lamda)

        UB = self.UB_matrix()
        s = UB @ hkl

        theta = np.zeros(two_theta.size)
        q = np.linalg.norm(s, axis=0)
        theta = np.arcsin(lamda*q/2)

        print('\nCalculated angle for new requested')
        print('(  H,  K,  L), 2theta omega chi phi')
        newphi = np.arctan2(s[1,:], s[0,:])
        newchi = np.arctan2(s[2,:], np.sqrt(s[0,:]**2+s[1,:]**2))
        newphi[s[0,:] < 0] += np.where(newphi[s[0,:] < 0] > 0, -np.pi, np.pi)
        fmt = '{:3.0f} {:3.0f} {:3.0f} {:5.2f} {:5.2f} {:5.2f} {:5.2f}'
        for i in range(q.size):
            angles = np.rad2deg([2*theta[i], theta[i], newchi[i], newphi[i]])
            print(fmt.format(hkl[0,i],hkl[1,i],hkl[2,i],*angles))

    def azimuthal_scan(self, h, k, l):

        hkl = np.array([h, k, l])

        UB = self.UB_matrix()

        s = UB @ hkl

        newphi = np.arctan2(s[1], s[0])
        if newphi > 0:
            if s[0] < 0:
                newphi -= np.pi
        else:
            if s[0] < 0:
                newphi += np.pi

        newchi = np.arctan2(s[2], np.sqrt(s[0]**2+s[1]**2))

        phi = newphi
        chi = newchi
        omega = 0

        R_phi = np.array([[np.cos(phi), np.sin(phi), 0],
                          [-np.sin(phi), np.cos(phi), 0],
                          [0, 0, 1]])

        R_chi = np.array([[np.cos(chi), 0, np.sin(chi)],
                          [0, 1, 0],
                          [-np.sin(chi), 0, np.cos(chi)]])

        R_omega = np.array([[np.cos(omega), np.sin(omega), 0],
                            [-np.sin(omega), np.cos(omega), 0],
                            [0, 0, 1]])

        R0 = R_omega @ R_chi @ R_phi

        psi = np.deg2rad(np.arange(0, 181, 10))

        print('{:8}{:8}{:8}{:8}'.format('psi', 'chi', 'phi', 'omega'))
        for i in range(psi.size):

            R_psi = np.array([[1, 0, 0],
                              [0, np.cos(psi[i]), np.sin(psi[i])],
                              [0, -np.sin(psi[i]), np.cos(psi[i])]])

            R = np.dot(R_psi, R0)

            psi_ = psi[i]
            chi_ = np.arctan2(np.sqrt(R[2,0]**2+R[2,1]**2), R[2,2])
            phi_ = np.arctan2(R[2,1], R[2,0])
            omega_ = np.arctan2(-R[1,2], R[0,2])
            angles = np.rad2deg([psi_, chi_, phi_, omega_])
            print('{:8.3f}{:8.3f}{:8.3f}{:8.3f}'.format(*angles))

    def angular_distance(self, setting1, setting2):

        return np.sqrt(sum((a-b)**2 for a, b in zip(setting1, setting2)))

    def greedy_sort(self, settings):

        n = len(settings)
        visited = np.zeros(n, dtype=bool)

        path = []
        current_index = 0
        path.append(current_index)
        visited[current_index] = True

        distance_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i,j] = self.angular_distance(settings[i],
                                                             settings[j])

        for _ in range(n-1):
            next_index = np.argmin([distance_matrix[current_index, j] if not visited[j] else np.inf for j in range(n)])
            visited[next_index] = True
            path.append(next_index)
            current_index = next_index

        return np.array([i for i in path])

    def generate_reflection_table(self, cif_file,
                                        max_reflections=None,
                                        k=[0,0,0],
                                        d_min=0.7):

        CreateSingleValuedWorkspace(OutputWorkspace='sample')

        LoadCIF(Workspace='sample', InputFile=cif_file)

        cs = mtd['sample'].sample().getCrystalStructure()
        ol = mtd['sample'].sample().getOrientedLattice()

        d_max = np.max([ol.a(), ol.b(), ol.c()])

        generator = ReflectionGenerator(cs)

        hkls = generator.getHKLsUsingFilter(d_min,
                                            d_max,
                                            ReflectionConditionFilter.Centering)

        satellite = np.linalg.norm(k) > 0

        hkls = [V3D(*(hkl+k)) for hkl in hkls]

        ds = np.array(generator.getDValues(hkls))
        F2s = np.array(generator.getFsSquared(hkls))*(not satellite)

        hkls = np.array(hkls)

        lamda = self.get_wavelength()

        B = self.B_matrix(*self.get_lattice_parameters())
        UB = self.UB_matrix()

        s = np.einsum('ij,kj->ik', B, hkls)

        q = np.sqrt(np.sum(s**2, axis=0))

        theta = np.arcsin(lamda*q/2)
        two_theta = 2*theta

        omega = two_theta/2

        s = np.einsum('ij,kj->ik', UB, hkls)

        phi = np.zeros_like(ds)
        chi = np.zeros_like(ds)

        for i in range(len(ds)):
            phi[i] = np.arctan(s[1,i]/s[0,i])
            if phi[i] > 0 and s[0,i] < 0:
                phi[i] -= np.pi
            elif phi[i] <= 0 and s[0,i] < 0:
                phi[i] += np.pi

            chi[i] = np.arctan2(s[2,i], np.sqrt(s[0,i]**2+s[1,i]**2))

        two_theta, omega, chi, phi = np.deg2rad([two_theta, omega, chi, phi])

        table = np.column_stack([*hkls, ds, F2s, two_theta, omega, chi, phi])

        return table

    def load_spice_data(self, filename):

        content = self.spice_data(filename)

        data, headertext, headers, \
        defxname, defyname, defxvalue, defyvalue = content

        ycol = headers.index('detector')
        theta2col = headers.index('s2')
        omegacol = headers.index('s1')
        chicol = headers.index('chi')
        phicol = headers.index('phi')
        moncol = headers.index('monitor')
        tempcol = headers.index('tsample')

        y = data[ycol,:]
        two_theta = data[theta2col,:]
        omega = data[omegacol,:]
        chi = data[chicol,:]
        phi = data[phicol,:]
        monitor = data[moncol,:]
        temp = data[tempcol,:]

        err = np.sqrt(y)

        y = y.flatten()
        err = err.flatten()
        two_theta = two_theta.flatten()
        omega = omega.flatten()
        chi = chi.flatten()
        phi = phi.flatten()
        monitor = monitor.flatten()
        temp = temp.flatten()

        err[err == 0] = 1

        data = np.column_stack([two_theta, omega, chi, phi, monitor, y, err, temp])

        xlab = str(defxname)
        ylab = str(defyname)

        return data, xlab, ylab

    def spice_data(self, filename):

        with open(filename, 'r') as f:
            data = []
            headertext = ''
            headers = []
            colcounter = False
            ncols = -1
            defxname = ''
            defyname = ''

            for line in f:
                commenttest = line.strip().split()

                if commenttest and commenttest[0] == '#':
                    if 'def_x' in commenttest:
                        defxname = commenttest[3]
                    if 'def_y' in commenttest:
                        defyname = commenttest[3]

                    if 'col_headers' in commenttest:
                        colcounter = True
                    elif colcounter:
                        headers = commenttest[1:]
                        colcounter = False

                    headertext += line

                else:
                    try:
                        tstring = np.array([float(x) for x in commenttest])
                    except ValueError:
                        continue

                    if ncols == -1:
                        ncols = len(tstring)
                        data.append(tstring)
                    else:
                        if len(tstring) == ncols:
                            data.append(tstring)
                        elif len(tstring) > ncols:
                            data.append(tstring[:ncols])
                        else:
                            ncols = len(tstring)
                            data = [row[:ncols] for row in data]
                            data.append(tstring)

        data = np.array(data).T

        try:
            defxvalue = headers.index(defxname)
            defyvalue = headers.index(defyname)
        except ValueError:
            defxvalue = None
            defyvalue = None

        content = data, headertext, headers, \
                  defxname, defyname, defxvalue, defyvalue

        return content

    def estimate_peak_width(self, x, y, err):

        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        err = np.asarray(err).flatten()

        data = np.column_stack((x, y, err))
        newdata = data[data[:, 0].argsort()]
        x = newdata[:, 0]
        y = newdata[:, 1]
        err = newdata[:, 2]

        data1 = np.column_stack((x, y))
        newdata = data1[data1[:,1].argsort()]
        y1 = np.sort(newdata[:,1])
        idx1 = round(len(y)/11)
        idx2 = round(len(y)/4)
        newx = newdata[idx1:idx2,0]
        newy = newdata[idx1:idx2,1]

        pin1 = np.polynomial.Polynomial.fit(newx, newy, deg=0)
        bkgrnd = np.mean(newy)
        slope = 0

        y = y-(slope*x+bkgrnd)

        dx = np.diff(x, prepend=x[0], append=x[-1])
        totarea = np.dot(dx, y)
        maxx = np.max(x)
        minx = np.min(x)
        av = totarea/(maxx-minx)
        xcom = np.mean(x)

        x2 = x[2:-2]
        y2 = y[2:-2]
        center = np.sum(x2*y2)/np.sum(y2)
        moment2 = np.sum((x2-center)**2*y2)/np.sum(y2)
        sigma = np.sqrt(moment2)*1.50

        idx = np.where(x <= center)[0]
        if len(idx) > 1 and len(idx) < len(x):
            ypeak = np.mean(y[len(idx)-1:len(idx)+2])
        else:
            ypeak = y[len(idx)]

        peak = ypeak-av

        newidx = np.where(y < av)[0]
        peakarea = totarea-np.dot(dx[newidx], y[newidx])*0.5
        area = peakarea*1.1

        width = sigma

        y3 = np.sort(y)
        newypeak = np.mean(y3[-3:])

        stdbk = np.std(y3[round(len(y)*0.10):round(len(y) * 0.35)])
        range_ = (maxx-minx) / 4.5
        centeridx = np.where(np.abs(x-center) < 0.20*range_)[0]
        peakheight = np.mean(y[centeridx])

        idx1 = np.where(np.abs(x-center) <= range_)[0]
        idx2 = np.where(np.abs(x-center) >= range_)[0]
        sigy = y[idx1]
        bky = y[idx2]

        ratio1 = np.std(sigy)/np.std(bky)
        ratio2 = peakheight/stdbk

        if ratio2 > 4 and np.abs(center-xcom) < 0.3:
            pass
        else:
            area = 0
            width = 2.5

        return bkgrnd, slope, area, center, width


    def gas_full(self, x, bkgrnd, slope, area, center, width):

        return bkgrnd+slope*(x-center)+area*np.exp(-((x-center)**2)/(2*width**2))

    def fit_peaks(self, scan_numbers, IPTS, exp, mcu=1):

        UB = self.UB_matrix()
        inv_UB = np.linalg.inv(UB)

        lamda = self.get_wavelength()
        chi0 = 0

        data_ub_refine = []
        data_struct_refine = []

        dirname = '/HFIR/{}/IPTS-{}/exp{}/Datafiles/'.format(self.instrument, IPTS, exp)

        for scannum in scan_numbers:

            fname = '{}_exp{}_scan{:04d}.dat'.format(self.instrument, exp, scannum)

            filename = os.path.join(dirname, fname)

            data = self.load_spice_data(filename)
            scanned_var = np.std(data[:,0:4], axis=0)
            # scanned_var = 1

            two_theta = -np.deg2rad(data[:,0])
            omega = -np.deg2rad(data[:,1]-data[:,0]/2)
            chi = np.deg2rad(data[:,2]-chi0)
            phi = np.deg2rad(data[:, 3])

            x = data[:,scanned_var]
            monitor = np.sort(data[:,4])[2:-2]
            monitorave = np.mean(monitor)
            ratio = monitorave/mcu
            y = data[:,5]/ratio
            err = data[:,6]/ratio

            theta2ave = np.mean(two_theta)
            omegaave = np.mean(omega)
            chiave = np.mean(chi)
            phiave = np.mean(phi)

            U1 = np.cos(omegaave)*np.cos(chiave)*np.cos(phiave)-np.sin(omegaave)*np.sin(phiave)
            U2 = np.cos(omegaave)*np.cos(chiave)*np.sin(phiave)+np.sin(omegaave)*np.cos(phiave)
            U3 = np.cos(omegaave)*np.sin(chiave)
            U = np.array([U1, U2, U3])
            save = 2*np.sin(theta2ave/2)/lamda*U
            hkl = inv_UB @ save

            have, kave, lave = np.round(hkl, 3)

            bkgrnd, slope, area, center, width = self.estimate_peak_width(x, y, err)

            initial_guess = [bkgrnd, slope, area, center, width]
            try:
                popt, pcov = scipy.optimize.curve_fit(self.gas_full, x, y, sigma=err, p0=initial_guess)
                perr = np.sqrt(np.diag(pcov))
            except RuntimeError:
                popt, perr = initial_guess, [0]*len(initial_guess)

            bestpa = popt
            bestdpa = perr

            intensity = bestpa[2]*np.sin(theta2ave)
            interr = bestdpa[2]*np.sin(theta2ave)
            width = abs(bestpa[4])
            widtherr = bestdpa[4]

            dxave = np.mean(np.diff(x))
            center = bestpa[3]-dxave / 2
            theta2ave = center

            plt.figure(1)
            plt.errorbar(x, y, yerr=err, fmt='ro')
            x_fit = np.linspace(min(x), max(x), len(x)*4)
            y_fit = self.gas_full(x_fit, *bestpa)
            plt.plot(x_fit, y_fit, 'k-')
            plt.title('Scan #{} ({},{},{})'.format(scannum,have,kave,lave))
            plt.xlabel('omega (degree)')
            plt.ylabel('counts')
            plt.show()

            angles = np.rad2deg([theta2ave, omegaave, chiave, phiave])
            theta2ave = np.rad2deg(theta2ave)

            data_ub_refine.append([have, kave, lave, *angles])
            data_struct_refine.append([scannum, have, kave, lave, theta2ave, intensity, interr, width, widtherr])

        return data_ub_refine, data_struct_refine


    # def generate_scan_macro(self, scan_list, two_theta_lim, chi_lim,
    #                         two_theta_blind, two_theta0=0,
    #                         omega0=0, chi0=0, phi0=0):

    #     UB = self.UB_matrix()
    #     B = self.B_matrix()

    #     lamda = self.get_wavelength()

    #     data = np.loadtxt(scan_list)
    #     h = data[:,0]
    #     k = data[:,1]
    #     l = data[: 2]
    #     intensity = data[:,3]

    #     hkl = np.row_stack([h, k, l])
    #     s = np.dot(B, hkl)

    #     q = np.linalg.norm(s, axis=0)
    #     theta = np.arcsin(lamda*q/2)
    #     two_theta = 2*theta
    #     omega = two_theta/2

    #     s = np.dot(UB, hkl)

    #     newphi = np.arctan2(s[1, :], s[0, :])
    #     newphi = np.where(s[0, :] < 0, newphi-np.pi, newphi)
    #     newphi = np.where(newphi < 0, newphi+np.pi, newphi)

    #     newchi = np.arctan2(s[2,:], np.sqrt(s[0,:]**2+s[1,:]**2))

    #     two_theta_min, two_theta_max = two_theta_lim
    #     chi_min, chi_max = chi_lim

    #     two_theta_al1, two_theta_al2, tol = two_theta_al

    #     idx = np.where(
    #         (two_theta >= two_theta_min) & (two_theta < two_theta_max) &
    #         (newchi > chi_min) & (newchi < chi_max) &
    #         (np.abs(two_theta-two_theta_ang1) > tol) &
    #         (np.abs(two_theta+two_theta_ang1) > tol) &
    #         (np.abs(two_theta-two_theta_ang2) > tol) &
    #         (np.abs(two_theta+two_theta_ang2) > tol)
    #     )

    #     angles = np.rad2deg([two_theta[idx], omega[idx], newchi[idx], newphi[idx]])

    #     table = np.vstack([h[idx], k[idx], l[idx], *angles, intensity[idx]]).T

    #     return table


