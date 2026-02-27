import numpy as np
from scipy import optimize as opt


def non_linear_triangulation(p1, p2, K, R, t):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))
    def get_initial_guess(P1, P2, pt1, pt2):
        A = np.vstack([pt1[0]*P1[2,:]-P1[0,:], pt1[1]*P1[2,:]-P1[1,:], pt2[0]*P2[2,:]-P2[0,:], pt2[1]*P2[2,:]-P2[1,:]])
        _, _, Vh = np.linalg.svd(A)
        X_h = Vh[-1]
        return X_h[:3] / X_h[3]
    X_init = get_initial_guess(P1, P2, p1, p2)
    def loss(X_opt, P1, P2, pt1, pt2):
        X_h = np.append(X_opt, 1)
        proj1 = P1 @ X_h
        proj2 = P2 @ X_h
        proj1 /= proj1[2]
        proj2 /= proj2[2]
        return np.sum((proj1[:2] - pt1[:2])**2) + np.sum((proj2[:2] - pt2[:2])**2)
    res = opt.minimize(loss, x0=X_init, args=(P1, P2, p1, p2), method='L-BFGS-B', tol=1e-8)
    return res.x