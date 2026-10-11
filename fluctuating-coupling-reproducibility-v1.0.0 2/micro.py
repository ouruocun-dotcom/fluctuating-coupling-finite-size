import numpy as np
import numba as nb
IID = True

@nb.njit(cache=True)
def orderp(th):
    c = 0.0; s = 0.0
    for j in range(th.size):
        c += np.cos(th[j]); s += np.sin(th[j])
    return c/th.size, s/th.size

@nb.njit(cache=True)
def run_one(N, K, D, tau, scheme, dt, Tb, T, every, seed):
    # scheme 0: Stratonovich white (Heun); 1: Ito white (Euler-Maruyama); 2: OU coloured noise (Heun)
    np.random.seed(seed)
    u = np.random.random()
    om = np.empty(N); th = np.empty(N)
    for j in range(N):
        om[j] = np.tan(np.pi*(np.random.random() - 0.5)) if IID else np.tan(np.pi*((j + u)/N - 0.5))
        th[j] = 2*np.pi*np.random.random()
    thp = np.empty(N); G0 = np.empty(N)
    nb_ = int(Tb/dt); nt = int(T/dt)
    out = np.empty(nt//every)
    sq = np.sqrt(2*D*dt)
    xi = np.random.normal()*np.sqrt(D/tau) if scheme == 2 else 0.0
    k = 0
    for n in range(nb_ + nt):
        c, s = orderp(th)
        if scheme == 2:
            a0 = (K + xi)*dt
            xin = xi + (-xi/tau)*dt + np.sqrt(2*D*dt)/tau*np.random.normal()
            a1 = (K + xin)*dt
            xi = xin
        else:
            a0 = K*dt + sq*np.random.normal(); a1 = a0
        for j in range(N):
            G0[j] = s*np.cos(th[j]) - c*np.sin(th[j])
            thp[j] = th[j] + om[j]*dt + G0[j]*a0
        if scheme == 1:
            for j in range(N):
                th[j] = thp[j]
        else:
            c1, s1 = orderp(thp)
            for j in range(N):
                G1 = s1*np.cos(thp[j]) - c1*np.sin(thp[j])
                th[j] = th[j] + om[j]*dt + 0.5*(G0[j]*a0 + G1*a1)
        if n >= nb_ and (n - nb_) % every == 0 and k < out.size:
            c, s = orderp(th)
            out[k] = c*c + s*s; k += 1
        if n % 1000 == 0:
            for j in range(N):
                th[j] = th[j] % (2*np.pi)
    return out

@nb.njit(parallel=True, cache=True)
def run_many(M, N, K, D, tau, scheme, dt, Tb, T, every, seed0):
    nt = int(T/dt)//every
    res = np.empty((M, nt))
    for m in nb.prange(M):
        res[m] = run_one(N, K, D, tau, scheme, dt, Tb, T, every, seed0 + m)
    return res
