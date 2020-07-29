"""
January 2020 Sebastian Kaltenbach
"""

import numpy as np



# Initialize terms corresponding to the thetas for faster computation
def setVarphi(n0,L,order):
    
    rea=n0.shape[1]
    b=n0.shape[0]
    nL=2*L+1
    dummy=np.ones((nL,nL))
    ind=np.tril_indices_from(dummy)
    
    if order==1:
        
        varphi=np.zeros((rea,b,nL))
        
        for r in range(rea):
            for k in range(b):
                for l in range(-L,L+1):
                    varphi[r,k,l+L]=n0[(k+l)%b,r]
                    
    elif order==2:
        
        varphi=np.zeros((rea,b,nL))
        varphib=np.zeros((rea,b,np.round(nL*(nL+1)/2).astype(int)))
        
        for r in range(rea):
            for k in range(b):
                for l in range(-L,L+1):
                    varphi[r,k,l+L]=n0[(k+l)%b,r]
                    
                dummy=np.outer(varphi[r,k],varphi[r,k])
                varphib[r,k]=dummy[ind]
        
        varphi=np.concatenate((varphi,varphib),axis=2)
    
    return varphi



def calcELBO(sigcd,tauab,res,sig,mt,n_rea,nbin,St,rrr):
    samples=200
    rea=n_rea
    s=np.ndarray((rea,nbin,samples))
    ELBO=0.
    
    ELBO-=np.sum(tauab[0]*np.log(tauab[1]))
    void,ldet=np.linalg.slogdet(St)
    ELBO+=0.5*ldet
    ELBO+=0.5*np.sum(np.log(np.exp(sig * -.5)))+np.sum(res)
    
    #sample from q(X)
    for r in range(rea):
        n1=np.exp(np.random.multivariate_normal(res[:,r],np.diag(np.exp(sig[:,r] * -.5)**2.),samples))
        s[r,:,:]=np.transpose(n1)
        
    #MC integrate softmax and virtual observable
    smp=0.
    for i in range(samples):
        for r in range(rea):
            smp-=(10**rrr)/2*(np.sum(s[r,:,i])-1.)**2    
            for k in range(nbin):
                smp+=mt[r,k]*np.log(s[r,k,i])
        
    smp/=samples   

    return ELBO+smp




def thetatraf(theta1,theta2,L):
    theta11=theta1.reshape(2*L+1)
    b_r_ind=np.tril_indices_from(theta2)
    mub=np.hstack((theta11,((2*theta2-np.diag(np.diag(theta2)))[b_r_ind])))
    
    return mub

def thetatraf2(thetat,L):
    theta=thetat.reshape(2*L*L+5*L+2)
    theta1=theta[0:(2*L+1)]
    theta1=theta1.reshape(2*L+1,1)
    b_ind=np.tril_indices(2*L+1)
    theta2=np.zeros((2*L+1,2*L+1))
    theta2[b_ind]=0.5*theta[2*L+1:]
    theta2=theta2+theta2.T
    
    
    return theta1,theta2



