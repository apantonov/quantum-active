import numpy as np

# =========================
# Parameters
# =========================
filename = "output"

tmax = 400.0
nb_trajs = 100
dt = 0.01
nk = 10			  # nk steps per interval dt
N = 24			   # dimension of matrices

tau = 10.0
D = 0.01

# Gain/loss rates; for the non-dissipative von Neumann equation, both are set to zero
nu_gain = 1e-8
nu_loss = 1e-2

nb_tsteps = int(tmax / dt)
tlist = np.arange(nb_tsteps + 1) * dt

# creation / annihilation operators
a = np.diag(np.sqrt(np.arange(1, N)), 1)
ad = a.T

x = (ad + a) / np.sqrt(2)
p = 1j * (ad - a) / np.sqrt(2)

H = p @ p / 2 + x @ x / 2

# =========================
# Initial state
# =========================
eigvals, V = np.linalg.eigh(H)
psi0 = V[:, [0]]			  # ground state
rho0 = psi0 @ psi0.conj().T
rho = rho0.copy()

# =========================
# AOUP trajectories
# =========================
dtt = dt / nk
nsteps = nk * len(tlist)

path = np.zeros((nsteps, nb_trajs))
pathv = np.zeros((nsteps, nb_trajs))

for k in range(nb_trajs):
	xc = 0.0
	vc = 0.0
	for m in range(nsteps):
		vc = vc * (1 - dtt / tau) + (np.sqrt(2 * D) / tau) * np.sqrt(dtt) * np.random.randn()
		xc = xc + dtt * vc
		path[m, k] = xc
		pathv[m, k] = vc

# =========================
# Time evolution
# =========================
erwx = np.zeros((len(tlist), nb_trajs), dtype=np.complex128)
erwx2 = np.zeros((len(tlist), nb_trajs), dtype=np.complex128)

x02 = np.trace(x @ x @ rho0).real
Id = np.eye(N)

for m in range(nb_trajs):
	rho = rho0.copy()

	for l in range(len(tlist)):
		erwx[l, m] = np.trace(x @ rho)
		erwx2[l, m] = np.trace(x @ x @ rho)

		for k in range(nk):
			idx = l * nk + k
			xc = path[idx, m] * Id

			# Hamiltonian
			H = p @ p / 2 + (x - xc) @ (x - xc) / 2

			# Dynamic Lindblad operators
			ac_dyn = (x - xc + 1j * p) / np.sqrt(2)
			acd_dyn = (x - xc - 1j * p) / np.sqrt(2)
			
			# Static Lindblad operators
			# ac_dyn = a
			# acd_dyn = ad


			# Predictor
			drho = (
				-1j * (H @ rho - rho @ H)
				+ 0.5 * nu_loss * (2 * ac_dyn @ rho @ acd_dyn
								   - acd_dyn @ ac_dyn @ rho
								   - rho @ acd_dyn @ ac_dyn)
				+ 0.5 * nu_gain * (2 * acd_dyn @ rho @ ac_dyn
								   - ac_dyn @ acd_dyn @ rho
								   - rho @ ac_dyn @ acd_dyn)
			)

			rho_pred = rho + drho * dtt
			rho_m = 0.5 * (rho + rho_pred)

			# Corrector
			drho_m = (
				-1j * (H @ rho_m - rho_m @ H)
				+ 0.5 * nu_loss * (2 * ac_dyn @ rho_m @ acd_dyn
								   - acd_dyn @ ac_dyn @ rho_m
								   - rho_m @ acd_dyn @ ac_dyn)
				+ 0.5 * nu_gain * (2 * acd_dyn @ rho_m @ ac_dyn
								   - ac_dyn @ acd_dyn @ rho_m
								   - rho_m @ ac_dyn @ acd_dyn)
			)

			rho = rho + drho_m * dtt

			#########################################
			# Agarwal dissipator parameters
			# nbar = nu_gain / (nu_loss - nu_gain)
			# kappa = 0.25 * (nu_loss - nu_gain)

			# ---------- Predictor ----------
			# rho_pred = (
			#	rho
			#	- 1j * (H @ rho - rho @ H) * dtt
			#	- 1j * kappa * (
			#		x @ p @ rho
			#		+ x @ rho @ p
			#		- p @ rho @ x
			#		- rho @ p @ x
			#	) * dtt
			#	- 2 * kappa * (nbar + 0.5) * (
			#		x @ x @ rho
			#		- 2 * x @ rho @ x
			#		+ rho @ x @ x
			#	) * dtt
			#)

			#rho_m = 0.5 * (rho + rho_pred)

			# ---------- Corrector ----------
			#rho = (
			#	rho
			#	- 1j * (H @ rho_m - rho_m @ H) * dtt
			#	- 1j * kappa * (
			#		x @ p @ rho_m
			#		+ x @ rho_m @ p
			#		- p @ rho_m @ x
			#		- rho_m @ p @ x
			#	) * dtt
			#	- 2 * kappa * (nbar + 0.5) * (
			#		x @ x @ rho_m
			#		- 2 * x @ rho_m @ x
			#		+ rho_m @ x @ x
			#	) * dtt
			#########################################
			#)


# =========================
# Expectation values
# =========================
MSD = np.mean(erwx2.real - x02, axis=1)
xav = np.mean(erwx.real, axis=1)

data = np.column_stack((tlist, MSD))
np.savetxt(f"MSD_{filename}.txt", data)
