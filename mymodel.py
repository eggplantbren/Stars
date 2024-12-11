import numpy as np

data = np.loadtxt("data.txt")
x, y, v, verr = data[:,0], data[:,1], data[:,2], data[:,3]


num_params = 10

def prior_transform(us):

    # Extract parameters. The first five are to do with the elliptical
    # gaussian density on the sky
    xc = -0.1 + 0.2*us[0]
    yc = -0.1 + 0.2*us[1]
    phi = 2.0*np.pi*us[2]
    q = us[3]
    L = 10.0**(-3.0 + 3.0*us[4])

    # The others are to do with the kinematics
    mu_v = -100.0 + 200.0*us[5]
    A_v = 10.0**(-1.0 +  4.0*us[6])
    L_v = 10.0**(-3.0 + 3.0*us[7])
    sig_v = 10.0**(-1.0 +  4.0*us[8])
    phi_v = 2.0*np.pi*us[9]

    return np.array([xc, yc, phi, q, L, mu_v, A_v, L_v, sig_v, phi_v])

def log_likelihood(params):

    xc, yc, phi, q, L, mu_v, A_v, L_v, sig_v, phi_v = params

    # This first term is to do with the elliptical gaussian density on the sky
    xx =  (x - xc)*np.cos(phi) + (y - yc)*np.sin(phi)
    yy = -(x - xc)*np.sin(phi) + (y - yc)*np.cos(phi)

    logl = 0.0
    logl += np.sum(-np.log(2.0*np.pi*L**2) \
                    - 0.5*((xx**2*q + yy**2/q)/L**2))

    # Now do the kinematics
    dist = x*np.sin(phi_v) - y*np.cos(phi_v)
    mu = mu_v + A_v*np.tanh(dist/L_v)

    var = sig_v**2 + verr**2
    logl += -0.5*np.sum(np.log(2.0*np.pi*var)) \
            - 0.5*np.sum((v - mu)**2/var)

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

