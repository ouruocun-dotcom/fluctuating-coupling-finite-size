"""Recompute moments, scaling, tails, Binder ratios and paired step differences."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
def density(N,K,D,ito=False,n=100001):
    r=np.linspace(1e-7,1-1e-7,n);g=r*(1-r*r)/2
    k=K-D/2*(1-r*r) if ito else K
    a=(k/2-1)*r-k/2*r**3+D*g*(1-3*r*r)/2+1/(2*N*r)
    b=1/N+2*D*g*g;f=2*a/b
    logp=-np.log(b)+np.r_[0,np.cumsum((f[1:]+f[:-1])*np.diff(r)/2)]
    p=np.exp(logp-logp.max());p/=np.trapezoid(p,r)
    return r,p
def theory(N,K,D,ito=False):
    r,p=density(N,K,D,ito)
    return np.array([np.trapezoid(r**m*p,r) for m in (1,2,4)])
def fit(ns,vs,rng):
    x=np.log(ns);w=(x-x.mean())/np.sum((x-x.mean())**2)
    est=-w@np.log([v.mean() for v in vs])
    means=np.stack([v[rng.integers(len(v),size=(2000,len(v)))].mean(axis=1) for v in vs])
    valid=(means>0).all(axis=0);bs=-(w[:,None]*np.log(means[:,valid])).sum(axis=0)
    return est,bs.std(ddof=1),*np.quantile(bs,[.025,.975])
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='results')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cs=json.loads((ROOT/'conditions.json').read_text());data=np.load(ROOT/'replica_statistics.npz',allow_pickle=False)
    rng=np.random.default_rng(2026101104)
    binder_rng=np.random.default_rng(2026101151)
    def select(**kw):return [c for c in cs if all(c.get(k)==v for k,v in kw.items())]
    def aggregate(group,key='moments'):
        return np.concatenate([data[c['id']+'_'+key][:,0] for c in group],axis=0)
    def initial(N,K,D):return sorted([c for c in select(batch='initial',N=N,K=K,D=D,scheme=0) if c['variant']!='step'],key=lambda c: c['variant']!='base')
    records=[]
    for c in cs:
        v=data[c['id']+'_moments']
        for branch in range(c['branches']):
            a=v[:,branch];u=1-a[:,2].mean()/(2*a[:,1].mean()**2)
            ids=binder_rng.integers(len(a),size=(10000,len(a)))
            ub=1-a[ids,2].mean(axis=1)/(2*a[ids,1].mean(axis=1)**2)
            se=a.std(axis=0,ddof=1)/np.sqrt(len(a))
            if c.get('kind')=='shared':
                group=a.reshape(5,8,3).mean(axis=1)
                se=group.std(axis=0,ddof=1)/np.sqrt(5)
            records.append([c['id'],branch,c['N'],c['K'],c['D'],c['scheme'],c['batch'],
                            *a.mean(axis=0),*se,u,*np.quantile(ub,[.025,.975])])
    with (out/'moments.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','branch','N','K','D','scheme','batch','M1','M2','M4','SE1','SE2','SE4','Binder','Binder_low','Binder_high']);w.writerows(records)
    with (out/'blocks.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','branch','block','mean_R2','SE_R2'])
        for c in cs:
            v=data[c['id']+'_blocks']
            for b in range(c['branches']):
                for j in range(6):w.writerow([c['id'],b,j+1,v[:,b,j].mean(),v[:,b,j].std(ddof=1)/np.sqrt(len(v))])
    with (out/'tails.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','branch','threshold','fraction','SE'])
        for c in cs:
            v=data[c['id']+'_tail']
            for b in range(c['branches']):
                for j,r in enumerate(data['thresholds']):w.writerow([c['id'],b,r,v[:,b,j].mean(),v[:,b,j].std(ddof=1)/np.sqrt(len(v))])
    fits=[]
    for D in (.8,1.6):
        groups=[initial(N,1.6,D) for N in (800,1600)]+[select(batch='followup',test='T4',N=N,D=D) for N in (3200,6400)]
        for key,column in [('moments',1),('tail',int(np.where(data['thresholds']==.3)[0][0]))]:
            vals=[aggregate(g,key)[:,column] for g in groups]
            fits.append(['large_size',1.6,D,0,key,*fit([800,1600,3200,6400],vals,rng)])
    for K,D,s,tau in sorted({(c['K'],c['D'],c['scheme'],c['tau']) for c in select(batch='repeat')}):
        groups=[select(batch='repeat',K=K,D=D,scheme=s,tau=tau,N=N) for N in (200,400,800,1600)]
        for j,m in enumerate((1,2,4)):
            fits.append(['repeat',K,D,s,str(m),*fit([200,400,800,1600],[aggregate(g)[:,j] for g in groups],rng)])
    with (out/'exponents.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample','K','D','scheme','observable','exponent','SE','low','high']);w.writerows(fits)
    paired=[]
    for c in select(batch='paired'):
        v=data[c['id']+'_moments'][:,:,1];ind=rng.integers(len(v),size=(10000,len(v)))
        for i,j in ((0,1),(2,3),(3,1)):
            d=v[:,i]-v[:,j];bs=d[ind].mean(axis=1)/v[ind,j].mean(axis=1)
            paired.append([c['N'],c['scheme'],i,j,100*d.mean()/v[:,j].mean(),* (100*np.quantile(bs,[.025,.975]))])
    with (out/'paired_steps.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['N','scheme','branch_i','branch_j','percent','low','high']);w.writerows(paired)
    # Compact review plots: same observables and data selections as the manuscript.
    plt.rcParams.update({'font.size':8,'legend.fontsize':7})
    def save(fig,name):fig.tight_layout();fig.savefig(out/(name+'.pdf'));plt.close(fig)
    def point(ax,group,col,style='o',color=None,scale=False):
        group=sorted(group,key=lambda c:c['N'])
        ns=sorted({c['N'] for c in group});vs=[aggregate([c for c in group if c['N']==n])[:,col] for n in ns]
        means=np.array([v.mean() for v in vs]);se=np.array([v.std(ddof=1)/np.sqrt(len(v)) for v in vs]);factor=np.array(ns) if scale else 1
        ax.errorbar(ns,factor*means,yerr=factor*se,fmt=style,color=color,ms=3)
    colors=['#555555','#2171b5','#238b45','#d94801','#756bb1']
    fig,ax=plt.subplots();k=np.linspace(0,2,201)
    for m in (1,2,4):ax.plot(k,2*(2-k)/m,label=f'm={m}')
    ax.set(xlabel='K',ylabel='D');ax.legend();save(fig,'fig1')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    for D,color in zip((0.,.2,.4,.8,1.6),colors):
        g=sum([initial(n,1.6,D) for n in (100,200,400,800,1600)],[])
        point(ax[0],g,1,color=color,scale=True)
        point(ax[0],select(batch='repeat',K=1.6,D=D,scheme=0),1,'^',color,True)
        point(ax[0],select(batch='followup',test='T4',D=D),1,'s',color,True)
        ns=np.geomspace(100,6400,30);ax[0].plot(ns,[n*theory(n,1.6,D)[1] for n in ns],color=color,label=f'D={D}')
    for K,D in ((1.,0.),(1.6,0.),(1.,1.6),(1.6,1.6)):
        rows=[r for r in fits if r[0]=='repeat' and r[1:4]==[K,D,0]]
        ax[1].errorbar([int(r[4]) for r in rows],[r[5] for r in rows],yerr=[r[6] for r in rows],fmt='o-',label=f'K={K},D={D}')
    ax[0].set(xscale='log',yscale='log',xlabel='N',ylabel='N mean R2');ax[1].set(xlabel='moment order',ylabel='exponent')
    for a in ax:a.legend()
    save(fig,'fig2')
    fig,ax=plt.subplots(1,2,figsize=(8,3));r=data['thresholds'];j=int(np.where(r==.3)[0][0])
    for D,color in zip((.2,.4,.8,1.6),colors[1:]):
        p=aggregate(initial(1600,1.6,D),'tail').mean(axis=0);ax[0].loglog(r,p,'.',color=color,label=f'D={D}')
        ns=(100,200,400,800,1600,3200,6400);xs=[];ys=[];es=[]
        for n in ns:
            g=initial(n,1.6,D) if n<=1600 else select(batch='followup',test='T4',N=n,D=D)
            if not g:continue
            v=aggregate(g,'tail')[:,j];xs.append(n);ys.append(v.mean());es.append(v.std(ddof=1)/np.sqrt(len(v)))
        ax[1].errorbar(xs,ys,yerr=es,fmt='o-',color=color)
    ax[0].set(xlabel='r',ylabel='Pr(R>r)');ax[1].set(xscale='log',yscale='log',xlabel='N',ylabel='time fraction R>0.3');ax[0].legend();save(fig,'fig3')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    for s,t in ((0,1.),(1,1.),(2,.1),(2,1.)):
        g=select(batch='repeat',K=1.6,D=1.6,scheme=s,tau=t);point(ax[0],g,1,scale=True)
    for test in ('T2a','T2b'):point(ax[1],select(batch='followup',test=test),1,scale=True)
    point(ax[1],select(batch='repeat',scheme=1),1,'v',scale=True)
    for a in ax:a.set(xscale='log',xlabel='N',ylabel='N mean R2')
    save(fig,'fig4')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    for n in (400,1600,6400):
        ks=np.linspace(.2,1.95,121);v=np.array([theory(n,k,.8) for k in ks]);ax[0].plot(ks,n*v[:,1],label=f'N={n}');ax[1].plot(ks,1-v[:,2]/(2*v[:,1]**2))
    ax[0].set(yscale='log',xlabel='K',ylabel='N mean R2');ax[1].set(xlabel='K',ylabel='Binder U');ax[0].legend();save(fig,'fig5')
    fig,ax=plt.subplots(1,2,figsize=(8,3))
    for K,D in ((1.,0.),(1.6,0.),(2.,0.),(1.6,.8),(1.,1.6)):
        target=ax[0 if D==0 else 1];ns=[];us=[]
        for n in (100,200,400,800,1600,3200,6400):
            if K==2:g=select(batch='critical',N=n)
            elif D==.8:g=initial(n,K,D) if n<=1600 else select(batch='followup',test='T4',N=n,D=D)
            else:g=select(batch='repeat',N=n,K=K,D=D,scheme=0)
            if not g:continue
            v=aggregate(g);ns.append(n);us.append(1-v[:,2].mean()/(2*v[:,1].mean()**2))
        target.plot(ns,us,'o-',label=f'K={K},D={D}')
    for a in ax:a.set(xscale='log',xlabel='N',ylabel='Binder U');a.legend()
    save(fig,'fig6')
    print('Wrote',out,'; 5 statistical tables and 6 review figures')
if __name__=='__main__':main()
