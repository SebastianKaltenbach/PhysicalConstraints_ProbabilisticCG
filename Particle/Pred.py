
"""
January 2020 Sebastian Kaltenbach
"""
import numpy as np

def thetatraf2(thetat,L):
    theta=thetat.reshape(2*L*L+5*L+2)
    theta1=theta[0:(2*L+1)]
    theta1=theta1.reshape(2*L+1,1)
    b_ind=np.tril_indices(2*L+1)
    theta2=np.zeros((2*L+1,2*L+1))
    theta2[b_ind]=0.5*theta[2*L+1:]
    theta2=theta2+theta2.T
    
    
    return theta1,theta2

def thetatraf(theta1,theta2,L):
    theta11=theta1.reshape(2*L+1)
    b_r_ind=np.tril_indices_from(theta2)
    mub=np.hstack((theta11,((2*theta2-np.diag(np.diag(theta2)))[b_r_ind])))
    
    return mub

def mufunc(a,b,n0,k,Dt,L,nbin):

    mu=0.
    mu+=n0[k]
    for i in range(-L,L+1):
       mu+=a[i+L]*Dt*n0[(k+i)%nbin]
       for j in range(-L,L+1):
           mu+=b[i+L,j+L]*n0[(k+i)%nbin]*n0[(k+j)%nbin]
            
    return mu

def prediction(res,sig,theta1,theta2,thetavar,t):
    data_pred=np.zeros((t,128,50))
    rho_pred=np.zeros((t,128,50))
    theta=thetatraf(theta1,theta2,8)
    
    for mc in range(50):
        nstart=res.reshape((128,1))
        thetamc=np.random.multivariate_normal(theta,thetavar)
        theta1mc,theta2mc= thetatraf2(thetamc,8)
        
        for k in range(128):
            data_pred[0,k,mc]=mufunc(theta1mc,theta2mc,nstart.reshape(128,1),k,1,8,128)
        rho_pred[0,:,mc]=np.random.multinomial(500000., data_pred[0,:,mc]/np.sum(data_pred[0,:,mc]))
    
        for j in range(1,t):
            for k in range(128):
                data_pred[j,k,mc]=mufunc(theta1mc,theta2mc,data_pred[j-1,:,mc],k,1,8,128)
            rho_pred[j,:,mc]=np.random.multinomial(500000., data_pred[j,:,mc]/np.sum(data_pred[j,:,mc]))
            
            
    rho_mean=np.mean(rho_pred/500000.,axis=2)
    rho_std=np.std(rho_pred/500000.,axis=2)
    #dp_05=np.quantile(data_pred,0.05,axis=2)
        
    return rho_mean,rho_std



