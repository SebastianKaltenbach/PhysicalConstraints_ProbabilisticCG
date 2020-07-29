
#Contains all necessary closed-form update laws


import numpy as np



# ARD -Pior
def tfunc(theta,thetavar,t,sigma,res,sig,Dt,nbin,L,varphi,b0):
    alpha=0.8
    
    a=t[0]
    
    b=b0*np.ones(t[1].shape[0])+0.5*(theta**2+np.diag(thetavar))
    
    return [alpha*t[0]+(1-alpha)*a,alpha*t[1]+(1-alpha)*b]


#Thetas (mean and variance)
def thetafunc(theta,thetavar,t,sigma,res,sig,Dt,nbin,L,varphi,n0):
    
    alpha=0.8
    
    b=nbin
    rea=varphi.shape[0]    
    
    sigb=np.zeros((2*L*L+5*L+2,2*L*L+5*L+2))
    y=np.zeros(2*L*L+5*L+2)
    
    sig2=np.exp(sig * -.5)
    phimu=np.exp(res+0.5*sig2**2)

    
    for r in range(rea):
        for k in range(b):
            
            sigb+=np.outer(varphi[r,k],varphi[r,k])
            y+=(phimu[k,r]-n0[k,r])*varphi[r,k]
    
    sigb*=sigma
    sigb+=np.diag(t[0]/t[1])   
    y*=sigma         
    
    mub=np.linalg.solve(sigb,y)
    
    return [alpha*theta+(1-alpha)*mub,alpha*thetavar+(1-alpha)*((np.linalg.inv(sigb)))]