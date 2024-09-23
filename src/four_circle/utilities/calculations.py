import numpy as np

import scipy.optimize
import scipy.linalg
import scipy.spatial

class FourCircle:

    def __init__(self, filename=None):

        self.a = self.b = self.c = self.alpha = self.beta = self.gamma = None

        self.lamda = None

        self.theta = self.phi = self.omega = None

        if filename is not None:

            self.read_parameters(filename)

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

        # lamda = self.get_wavelength()

        s = UB @ hkl
        # q = np.linalg.norm(Q)
        # theta = np.arcsin(lamda*q/2)
        
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