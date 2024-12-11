import numpy as np

data = np.loadtxt("data.txt")
x, y, v, verr = data[:,0], data[:,1], data[:,2], data[:,3]


num_params = 5

def prior_transform(us):

    # Extract parameters. The first five are to do with the elliptical
    # gaussian density on the sky
    xc = -0.1 + 0.2*us[0]
    yc = -0.1 + 0.2*us[1]
    theta = 2.0*np.pi*us[2]
    q = us[3]
    L = 10.0**(-3.0 + 3.0*us[4])

    # The others are to do with the kinematics
    # Not implemented yet

    return np.array([xc, yc, theta, q, L])

def log_likelihood(params):

    xc, yc, theta, q, L = params

    # This first term is to do with the elliptical gaussian density on the sky
    xx =  (x - xc)*np.cos(theta) + (y - yc)*np.sin(theta)
    yy = -(x - xc)*np.sin(theta) + (y - yc)*np.cos(theta)

    logl = np.sum(-np.log(2.0*np.pi*L**2) \
                    - 0.5*((xx**2*q + yy**2/q)/L**2))

    return logl


def both(us):
    return log_likelihood(prior_transform(us))

