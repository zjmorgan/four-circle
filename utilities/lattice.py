import numpy as np

import scipy.optimize
import scipy.linalg
import scipy.spatial

class Lattice:

    def __init__(self):

        self.a = self.b = self.c = self.alpha = self.beta = self.gamma = None

        self.lamda = None

        self.theta = self.phi = self.omega = None

    def get_parameters(self):

        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def get_wavelength(self):

        return self.lamda

    def set_wavelength(self, lamda):

        self.lamda = lamda

    def get_axis_angles(self):

        return self.theta, self.phi, self.omega 

    def set_axis_angles(self, values):

        self.theta, self.phi, self.omega = values

    def set_parameters(self, params):

        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = params

    def read_parameters(self, filename):

        params = []
        with open(filename, 'r') as f:
            for line in f:
                values = line.split()
                for value in values:
                    params.append(float(value))
                    if len(params) == 6:
                        self.set_parameters(params)
                    elif len(params) == 7:
                        self.set_wavelength(value)

    def write_parameters(self, filename):

        params = self.get_parameters()        

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

    def get_orientation_angles(self, U):

        omega = np.arccos((np.trace(U)-1)/2)

        val, vec = np.linalg.eig(U)

        ux, uy, uz = vec[:,np.argwhere(np.isclose(val, 1))[0][0]].real

        theta = np.arccos(uz)
        phi = np.arctan2(uy, ux)

        return phi, theta, omega

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

        B = self.B_matrix(*self.get_parameters())
        U = self.U_matrix(*self.get_orientation())

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

    def residual(self, x, hkl, Q, fun):

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U,B)

        return (np.einsum('ij,lj->li', UB, hkl)*2*np.pi-Q).flatten()

    def optimize_lattice(self, cell='Triclinic'):

        a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

        phi, theta, omega = self.get_orientation_angles()

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
        args = (self.hkl, self.Q, fun)

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
