import numpy as np
import numba as nb

@nb.njit(cache=True)
def step(th,om,h,dw,scheme,split,K):
    if split:th=th+.5*h*om
    c=np.cos(th).mean();s=np.sin(th).mean();g=s*np.cos(th)-c*np.sin(th)
    free=0.*om if split else om;pred=th+h*(free+K*g)+dw*g
    if scheme==0:
        cp=np.cos(pred).mean();sp=np.sin(pred).mean();gp=sp*np.cos(pred)-cp*np.sin(pred)
        th=th+h*free+.5*(K*h+dw)*(g+gp)
    else:th=pred
    if split:th=th+.5*h*om
    return th%(2*np.pi)

@nb.njit(cache=True)
def pair(seed,scheme):
    np.random.seed(seed);N=400
    om=np.tan(np.pi*(np.random.random(N)-.5));initial=2*np.pi*np.random.random(N)
    th=np.empty((4,N))
    for b in range(4):th[b]=initial
    out=np.empty((4,30000));idx=0
    for n in range(160000):
        d1=np.sqrt(.032)*np.random.normal();d2=np.sqrt(.032)*np.random.normal()
        for split in range(2):
            b=2*split
            th[b]=step(th[b],om,.02,d1+d2,scheme,split,1.6)
            th[b+1]=step(th[b+1],om,.01,d1,scheme,split,1.6)
            th[b+1]=step(th[b+1],om,.01,d2,scheme,split,1.6)
        if n>=10000 and (n+1-10000)%5==0:
            for b in range(4):
                c=np.cos(th[b]).mean();s=np.sin(th[b]).mean();out[b,idx]=c*c+s*s
            idx+=1
    return out

@nb.njit(parallel=True,cache=True)
def pairs(scheme):
    out=np.empty((16,4,30000))
    for m in nb.prange(16):out[m]=pair(2400000000+64*scheme+m,scheme)
    return out

@nb.njit(cache=True)
def critical_one(N,seed):
    np.random.seed(seed);om=np.tan(np.pi*(np.random.random(N)-.5));th=2*np.pi*np.random.random(N)
    out=np.empty(30000);idx=0
    for n in range(160000):
        th=step(th,om,.02,0.,0,0,2.)
        if n>=10000 and (n+1-10000)%5==0:
            c=np.cos(th).mean();s=np.sin(th).mean();out[idx]=c*c+s*s;idx+=1
    return out

@nb.njit(parallel=True,cache=True)
def critical(N,index):
    out=np.empty((32,30000))
    for m in nb.prange(32):out[m]=critical_one(N,2410000000+64*index+m)
    return out

@nb.njit(cache=True)
def advance(th,om,h,dw,scheme,split):
    if split:th=th+.5*h*om
    c=np.cos(th).mean();s=np.sin(th).mean()
    g=s*np.cos(th)-c*np.sin(th)
    f=1.6*g
    free=0.*om if split else om
    pred=th+h*(free+f)+dw*g
    if scheme==0:
        cp=np.cos(pred).mean();sp=np.sin(pred).mean()
        gp=sp*np.cos(pred)-cp*np.sin(pred)
        th=th+h*free+.5*h*1.6*(g+gp)+.5*dw*(g+gp)
    else:th=pred
    if split:th=th+.5*h*om
    return th%(2*np.pi)

@nb.njit(cache=True)
def one(seed,scheme):
    np.random.seed(seed)
    om=np.tan(np.pi*(np.random.random(1600)-.5))
    initial=2*np.pi*np.random.random(1600)
    th=np.empty((4,1600))
    for b in range(4):th[b]=initial
    out=np.empty((4,30000));idx=0
    for n in range(160000):
        d1=np.sqrt(.032)*np.random.normal();d2=np.sqrt(.032)*np.random.normal()
        for split in range(2):
            b=2*split
            th[b]=advance(th[b],om,.02,d1+d2,scheme,split)
            th[b+1]=advance(th[b+1],om,.01,d1,scheme,split)
            th[b+1]=advance(th[b+1],om,.01,d2,scheme,split)
        if n>=10000 and (n+1-10000)%5==0:
            for b in range(4):
                c=np.cos(th[b]).mean();s=np.sin(th[b]).mean();out[b,idx]=c*c+s*s
            idx+=1
    return out,np.max(np.abs(om))

@nb.njit(parallel=True,cache=True)
def many(scheme):
    out=np.empty((16,4,30000));ommax=np.empty(16)
    for m in nb.prange(16):out[m],ommax[m]=one(2300000000+64*scheme+m,scheme)
    return out,ommax
