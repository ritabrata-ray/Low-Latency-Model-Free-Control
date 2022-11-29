import numpy as np
import obstacle_2D_sys

hat_lamda_1=0.2 #Oracle estimate of parameters
hat_lamda_2=-0.1 #Oracle estimate of parameters

def correction_controller(sys, u_last, u, theta=0.1):  # See algorithm 2 from paper
    gm = np.zeros(2)
    # ccheck=0
    u_corr = np.zeros(2)
    g_hat = np.array([[(1 / hat_lamda_1), 0], [0, (1 / hat_lamda_2)]])
    x_corr = np.zeros(2)
    xd = sys.state_derivative()
    gphi = sys.grad_phi()
    if (sys.phi_val() < theta):
        dot_phi = np.dot(gphi, xd)
        if (dot_phi >= 0):
            a = u_last
        else:
            if (gphi[0] * xd[0] < 0):
                gm[0] = 1
            if (gphi[1] * xd[1] < 0):
                gm[1] = 1
            x_corr[0] = gm[0] * xd[0]
            x_corr[1] = gm[1] * xd[1]
            u_corr = u_last - np.multiply(np.matmul(g_hat, x_corr), 2)
            a = u_corr
            # print("Correction used")
            # ccheck=1
    else:
        a = u
    return a


def recovery_controller(sys, u_last, u, theta=0.1, eta=0.1):
    gm = np.zeros(2)
    # ccheck=0
    u_rec = np.zeros(2)
    u_corr = np.zeros(2)
    g_hat = np.array([[(1 / hat_lamda_1), 0], [0, (1 / hat_lamda_2)]])
    x_corr = np.zeros(2)
    xd = sys.state_derivative()
    gphi = sys.grad_phi()
    if (sys.phi_val() <= theta):
        dot_phi = np.dot(gphi, xd)
        if ((sys.phi_val()==theta) and (dot_phi >= 0)):
            a = u_last
        elif ((sys.phi_val()==theta) and (dot_phi < 0)):  #correction controller case
            if (gphi[0] * xd[0] < 0):
                gm[0] = 1
            if (gphi[1] * xd[1] < 0):
                gm[1] = 1
            x_corr[0] = gm[0] * xd[0]
            x_corr[1] = gm[1] * xd[1]
            u_corr = u_last - np.multiply(np.matmul(g_hat, x_corr), 2)
            a = u_corr
        else:  # recovery case
            norm_grad_phi_square = gphi[0]**2 + gphi[1]**2
            n_d = ( (-1*eta) / (norm_grad_phi_square) )
            g=sys.get_g_x()
            u_rec=u_last-np.matmul( np.linalg.pinv(g), (xd+np.multiply(gphi,n_d)) )
            a = u_rec
    else:
        a = u  #default case use nominal controller
    return a
