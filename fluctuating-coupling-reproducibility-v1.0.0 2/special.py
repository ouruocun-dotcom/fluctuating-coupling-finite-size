import numpy as np
import numba as nb

@nb.njit(cache=True)
def drift(th,K,D,variant):
    c=np.cos(th).mean();s=np.sin(th).mean();r2=c*c+s*s
    g=s*np.cos(th)-c*np.sin(th)
    h=c*np.cos(th)+s*np.sin(th)
    a=K*g
    if variant>0:a-=D/2*(1-r2)*g
    if variant==2:a+=D*g*h
    return a,g

@nb.njit(cache=True)
def special_one(N,K,D,tau,dt,burn,T,seed,kind,frequency_seed):
    np.random.seed(frequency_seed if frequency_seed>=0 else seed)
    # Preserve supplied iid draws for special integrator comparisons.
    u=np.random.random();om=np.empty(N);th=np.empty(N)
    for j in range(N):
        om[j]=np.tan(np.pi*(np.random.random()-.5))
        th[j]=2*np.pi*np.random.random()
    if frequency_seed>=0:
        np.random.seed(seed)
        th=2*np.pi*np.random.random(N)
    every=round(.1/dt);nburn=round(burn/dt);nsteps=round(T/dt)
    out=np.empty(nsteps//every);k=0
    xi=np.random.normal()*np.sqrt(D/tau) if kind==4 else 0.
    decay=np.exp(-dt/(2*tau));sig=np.sqrt(D/tau*(1-decay*decay))
    for n in range(nburn+nsteps):
        if kind==4:
            xm=decay*xi+sig*np.random.normal();xn=decay*xm+sig*np.random.normal()
            a,_=drift(th,K+xi,0.,0);k1=om+a
            a,_=drift(th+.5*dt*k1,K+xm,0.,0);k2=om+a
            a,_=drift(th+.5*dt*k2,K+xm,0.,0);k3=om+a
            a,_=drift(th+dt*k3,K+xn,0.,0);k4=om+a
            th+=dt*(k1+2*k2+2*k3+k4)/6;xi=xn
        else:
            dw=np.sqrt(2*D*dt)*np.random.normal()
            a,g=drift(th,K,D,kind)
            pred=th+om*dt+a*dt+g*dw
            ap,gp=drift(pred,K,D,kind)
            th+=om*dt+.5*(a+ap)*dt+.5*(g+gp)*dw
        if n>=nburn and (n-nburn)%every==0:
            c=np.cos(th).mean();s=np.sin(th).mean();out[k]=c*c+s*s;k+=1
        if n%1000==0:th%=2*np.pi
    return out

@nb.njit(parallel=True,cache=True)
def special_many(M,N,K,D,tau,dt,burn,T,seed,kind,shared):
    out=np.empty((M,round(T/dt)//round(.1/dt)))
    for m in nb.prange(M):
        freqseed=seed+10000+(m//8)*100 if shared else -1
        out[m]=special_one(N,K,D,tau,dt,burn,T,seed+m,kind,freqseed)
    return out

@nb.njit(cache=True)
def meso_one(N,K,D,tau,dt,burn,T,seed):
    np.random.seed(seed)
    x=np.random.normal()/np.sqrt(2*N);y=np.random.normal()/np.sqrt(2*N)
    xi=np.random.normal()*np.sqrt(D/tau)
    decay=np.exp(-dt/tau);sig=np.sqrt(D/tau*(1-decay*decay))
    every=round(.1/dt);nburn=round(burn/dt);nsteps=round(T/dt)
    out=np.empty(nsteps//every);k=0
    for n in range(nburn+nsteps):
        xn=decay*xi+sig*np.random.normal()
        dx=np.sqrt(dt/N)*np.random.normal();dy=np.sqrt(dt/N)*np.random.normal()
        f=-1+(K+xi)/2*(1-x*x-y*y)
        xp=x+f*x*dt+dx;yp=y+f*y*dt+dy
        fp=-1+(K+xn)/2*(1-xp*xp-yp*yp)
        x+=.5*(f*x+fp*xp)*dt+dx
        y+=.5*(f*y+fp*yp)*dt+dy;xi=xn
        if n>=nburn and (n-nburn)%every==0:out[k]=x*x+y*y;k+=1
    return out

@nb.njit(parallel=True,cache=True)
def meso_many(M,N,K,D,tau,dt,burn,T,seed):
    out=np.empty((M,round(T/dt)//round(.1/dt)))
    for m in nb.prange(M):out[m]=meso_one(N,K,D,tau,dt,burn,T,seed+m)
    return out
