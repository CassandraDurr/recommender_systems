import numpy as np
import matplotlib.pyplot as plt

def normal_prob(x:float, mu:float, sigma:float)->float:
    prob = 1/(sigma*np.sqrt(2*np.pi))
    prob *= np.exp(-0.5*(((x-mu)/sigma)**2))
    return prob
    
with open("param/u_matrix.npy", 'rb') as f:
    U_mat = np.load(f)
with open("param/v_matrix.npy", 'rb') as f:
    V_mat = np.load(f)

# Plotting p(r, u, v)
def plot_prob_ruv(R:float, U:np.array, V:np.array, lmd:float, tau:float)->None :
    # print(f"U: min = {np.min(U)}, max = {np.max(U)}\nV: min = {np.min(V)}, max = {np.max(V)}")
    u_vals = np.linspace(-5, 5, num=100)
    v_vals = np.linspace(-5, 5, num=100)
    # u_vals = np.linspace(np.min(U)-0.01, np.max(U)+0.01, num=100)
    # v_vals = np.linspace(np.min(V)-0.01, np.max(V)+0.01, num=100)
    U_mesh, V_mesh = np.meshgrid(u_vals, v_vals)
    
    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(U_mesh.shape)
    for i in range(U_mesh.shape[0]):
        for j in range(U_mesh.shape[1]):
            val = normal_prob(R, U_mesh[i,j]*U_mesh[i,j], np.sqrt(1/lmd))
            val *= normal_prob(U_mesh[i,j], 0.0, np.sqrt(1/tau))
            val *= normal_prob(V_mesh[i,j], 0.0, np.sqrt(1/tau))
            pdf[i,j] = val

    # Plotting contour plots
    plt.figure(figsize=(9, 6))
    plt.contourf(U_mesh, V_mesh, pdf, cmap='viridis')
    plt.xlabel("U")
    plt.ylabel("V")
    plt.title(f'p(R={R},U,V) with tau={tau} and lambda={lmd}')
    # plt.savefig(f"figures/p_ruv_{R}_{tau}_{lmd}.png")
    plt.show()
    
# Use this to produce plots
# for r_value in [1.0, 3.0, 5.0, 7.0, 9.0] :
#     plot_prob_ruv(r_value, U_mat, V_mat, 0.15, 0.01)
for tau in [0.001, 0.01, 0.05, 0.1, 0.5, 1] :
    for lmd in [0.001, 0.01, 0.05, 0.1, 0.5, 1] :
        plot_prob_ruv(1.0, U_mat, V_mat, lmd, tau)