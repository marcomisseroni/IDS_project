import numpy as np

# measurement covariance
R_rr = np.identity(3) * 0.01
R_rp = np.identity(2) * 0.01
# model covariance
Q = np.identity(3) * 0.001
# time step
dt = 0.1
# person model covariance
Q_p = np.zeros((4, 4))
Q_p[0, 0] = dt**4 / 4
Q_p[0, 2] = dt**3 / 2
Q_p[1, 1] = dt**4 / 4
Q_p[1, 3] = dt**3 / 2
Q_p[2, 0] = dt**3 / 2
Q_p[2, 2] = dt**2
Q_p[3, 1] = dt**3 / 2
Q_p[3, 3] = dt**2