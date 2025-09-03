import matplotlib
# matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt


xl, xr = 0, 6e3
X = np.linspace(xl,xr,800)
R_0 = 1/12
V_0 = 60
L_0 = 50
R0 = (np.full(800, R_0))
# R3 = np.linspace(R_0,1/7,1000)
R1 = np.hstack((np.linspace(R_0,1/11,400), np.linspace(1/11,R_0,400)))
R2 = np.hstack((np.linspace(R_0,1/9,400), np.linspace(1/9,R_0,400)))
R3 = np.hstack((np.linspace(R_0,1/7,400), np.linspace(1/7,R_0,400)))

V0 = V_0 * np.exp(-R0*X/L_0)
V1 = V_0 * np.exp(-R1*X/L_0)
V2 = V_0 * np.exp(-R2*X/L_0)
V3 = V_0 * np.exp(-R3*X/L_0)

I0 = (V_0 / R0) * (1 - np.exp(-R0*X/L_0))
I1 = (V_0 / R1) * (1 - np.exp(-R1*X/L_0))
I2 = (V_0 / R2) * (1 - np.exp(-R2*X/L_0))
I3 = (V_0 / R3) * (1 - np.exp(-R3*X/L_0))


plt.plot(X,V0, label='12Mho')
plt.plot(X,V1, label='11Mho')
plt.plot(X,V2, label='9Mho')
plt.plot(X,V3, label='7Mho')
# plt.hlines(I0[400], xl, xr, color='C0', linestyle='dashed')
# plt.hlines(I1[400], xl, xr, color='C1', linestyle='dashed')
# plt.hlines(I2[400], xl, xr, color='C2', linestyle='dashed')
# plt.hlines(I3[400], xl, xr, color='C3', linestyle='dashed')
# plt.fill_between(X, I0, I3, color='none', hatch='/', edgecolor='C3')
plt.legend()
# plt.ylabel('kA')
# plt.gca().set_yticklabels(range(-100,800,100))
# plt.show()
plt.savefig("figures_2021_Special/model.png")