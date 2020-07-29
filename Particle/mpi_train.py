"""
January 2020 Sebastian Kaltenbach

Code for the particle example.
"""

#The SVI is parallelized using MPI. Current setting calculates 2 samples per node.
from mpi4py import MPI
from threading import Thread

import tensorflow as tf
import numpy as np
import scipy.io
import pickle

import ParaOpt2 as opt
import Utilities2 as ut

comm = MPI.COMM_WORLD
size = comm.Get_size()

rank = comm.Get_rank()


#SVI to compute mean and variance for the CG state variables
def estep(sig_start,res_start,n0,m1,theta1,theta2,sigma,nbin,L,processed,processed2,inn,counter,rrr,steps):
    #Initilaise penalty constant and parameters as tf placeholders
    r = tf.placeholder(tf.float64, shape=[])
    theta_tf = tf.placeholder(tf.float64, shape=(2*L+1,1))
    theta_tf2 = tf.placeholder(tf.float64, shape=(2*L+1,2*L+1))
    sigma_tf = tf.placeholder(tf.float64, shape=(1,))
    config = tf.ConfigProto(intra_op_parallelism_threads=1, 
                        inter_op_parallelism_threads=1)
    
    #Compute parametrized update law
    def mufunc(a,b,n0,k,Dt,L,nbin):

        mu=0.
        mu+=n0[k]
        ind=np.linspace((k-L),(k+L),2*L+1,dtype=int)%nbin
        mu+=Dt*tf.transpose(a)@n0[ind]
        mu+=tf.transpose(n0[ind])@b@n0[ind]
            
        return mu

    # Loss function to bve optimized for SVI. 
    #Further vectorization for speedup possible
    def loss(theta,b,n0,x,p,sigma,m,Dt,L,nbin):
    
        loss=0.
    
        for i in range(0,nbin):
            loss-= sigma/2*(x[i]-mufunc(theta,b,n0,i,Dt,L,nbin))**2              
        
        for i in range(0,nbin):
            loss+=m[i]*tf.log(x[i])
           
        # Virtual Observable for the mass constraint
        loss-=p/2*(tf.reduce_sum(x,axis=0)-1.)**2
     
        
        return loss

    def reparameterize( mean, logvar):
        eps = tf.random.normal(mean.shape,dtype=tf.float64)
        return eps * tf.exp(logvar * -.5) + mean

    steplength=0.001*0.96**(counter-1)
    steplength=max(steplength,0.00005)
    x = tf.Variable(res_start,dtype=tf.float64, trainable=True) #Mean
    delta = tf.Variable(sig_start,dtype=tf.float64, trainable=True) #Variance
    y=tf.exp(reparameterize(x,delta))
    f_x = tf.reduce_sum(loss(theta_tf,theta_tf2,n0,y,r,sigma_tf,m1,1.0,L,nbin))+tf.reduce_sum(-0.5*delta)+tf.reduce_sum(x)
    lossv = -f_x
    optimizer = tf.train.AdamOptimizer(steplength)
    opt = optimizer.minimize(lossv)
    reset_optimizer_op = tf.variables_initializer(optimizer.variables())
    tttt=optimizer._get_beta_accumulators()

    with tf.Session(config=config) as sess:#
        sess.run(tf.global_variables_initializer())
        for iter in range(1):                
            sess.run(reset_optimizer_op)
            sess.run(tf.variables_initializer(tttt))
            for s in range(1,rrr):
                for i in range(steps):
                    sess.run(opt,feed_dict={r:10**s,theta_tf:theta1,theta_tf2:theta2,sigma_tf:sigma})

            res=sess.run(x,feed_dict={r: 0.,theta_tf:theta1,theta_tf2:theta2,sigma_tf:sigma})
            sig=sess.run(delta,feed_dict={r: 0.,theta_tf:theta1,theta_tf2:theta2,sigma_tf:sigma})

    processed[inn,:]=res.T
    processed2[inn,:]=sig.T
    
    
    
n_rea=2*size  #2 samples per node
nbin=128 # Dimension of CG states (64 for AD)
Dt=1.
L=8 #InteractionLength
theta1=np.ones((2*L+1,1))*0.0  #First order interaction coefficients
theta2=np.ones((2*L+1,2*L+1))*0.0 #Second order interaction coefficients

sigma=np.ones((1,))*0.001

# Main computation is done on rank 0. Therefore hyperpriors are here initialized
if rank==0:
    c0=0.0000000001 #Sparsity parameter
    theta=np.ones((2*L*L+5*L+2,1))*0.0
    tau=np.zeros((2*L*L+5*L+2))
    sigcd=np.zeros(2)
    sigcd[0]=n_rea*nbin*(0.5)+0.0000000001
    sigcd[1]=sigcd[0]/sigma
    
    at=(c0+1/2)*np.ones(2*L*L+5*L+2)
    bt=c0*np.ones(2*L*L+5*L+2)+0.5*(theta[0]**2)
    t=[at,bt]  # ARD Prior
    
    save=np.zeros((2*L*L+5*L+2,500))
    thetavar=np.ones((2*L*L+5*L+2,2*L*L+5*L+2))*0.1

sendbuf = None
sendbuf2 = None
if rank == 0:
     mat = scipy.io.loadmat('Data_Burger.mat')#or Data_AD 
     n=mat["n"]
     m=mat["m"]
     #Initialize SVI
     n0=n[0,0:size*2,:].reshape(size*2,nbin)
     varphi=ut.setVarphi(n0.T,L,2)
     m1=m[1,0:size*2,:].reshape(size*2,nbin)
     sendbuf = np.empty([size*2, nbin], dtype='d')
     sendbuf2 = np.empty([size*2, nbin], dtype='d')
     for i in range(size*2):
         sendbuf[i,:]=n0[i,:]
         sendbuf2[i,:]=m1[i,:]
         
#Dsitribute samples to nodes
recvbuf = np.empty([2,nbin], dtype='d')
recvbuf2 = np.empty([2,nbin], dtype='d')
comm.Scatter(sendbuf, recvbuf, root=0)
comm.Scatter(sendbuf2, recvbuf2, root=0)
m1_local=recvbuf2
n0_local=recvbuf
recvbuf=np.log(recvbuf)
d0=np.log(np.ones((nbin,2))*0.01)*(-2)
recvbuf2=d0.T
processed = np.empty([2,nbin], dtype='d')
processed2 = np.empty([2,nbin], dtype='d')
counter=1

for iter in range(500):
    
    rrr=11
    steps=300
    if iter<3:
        steps=1200
        
    for i in range(2):
        th = Thread(target=estep, args=(recvbuf2[i,:].reshape(nbin,1),recvbuf[i,:].reshape(nbin,1),n0_local[i,:].reshape(nbin,1),m1_local[i,:].reshape(nbin,1),theta1,theta2,sigma,nbin,L,processed,processed2,i,counter,rrr,steps))
        th.start()
        th.join()

    # Collect everything at rank 0 node
    recvbuf = None
    recvbuf2 = None
    if rank == 0:
        recvbuf = np.empty([size*2, nbin], dtype='d')
        recvbuf2 = np.empty([size*2, nbin], dtype='d')
    comm.Gather(processed, recvbuf, root=0)
    comm.Gather(processed2, recvbuf2, root=0)



    if rank == 0:
        theta=ut.thetatraf(theta1,theta2,L)
        res=recvbuf.T  #mean-parameter of CG states
        sig=recvbuf2.T  #std-parameter of CG states
        save[:,iter]=theta  #Convergence of thetas with iterations
        
        u_steps=15
        if iter<3:
            u_steps=150
            
        for u in range(u_steps):
    
        # compute theta with closed form update
            theta,thetavar=opt.thetafunc(theta,thetavar,t,sigma,res,sig,Dt,nbin,L,varphi,n0.T)
        
        #compute ARD prior with closed form update
            t=opt.tfunc(theta,thetavar,t,sigma,res,sig,Dt,nbin,L,varphi,c0)
        
        #Parameter sigma (the variance of the virtual observable of the residual) is decreased to the desired value:
        
            sigma=np.exp(15.)*(1-np.exp(-iter/100))
            if iter>400:
                sigma=np.exp(15.)
            sigma=sigma.reshape((1,))
        
        theta1,theta2=ut.thetatraf2(theta,L)
        
        # ELBO
        #elbo[0,iter]=ut.calcELBO(sigcd,t,res,sig,m1,n_rea,nbin,thetavar,rrr)
    
    
    if rank == 0:
        tmp=0.
    else:
        theta1 = np.empty((2*L+1,1),dtype='d')
        theta2 = np.empty((2*L+1,2*L+1),dtype='d')
        sigma = np.empty(1,dtype='d')
    comm.Bcast(theta1, root=0)
    comm.Bcast(theta2, root=0)
    comm.Bcast(sigma, root=0)
    
    recvbuf=processed
    recvbuf2=processed2
    counter=counter+1 

    #Save results
if rank==0 :   
    with open('Results_Burgers.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([theta1, theta2, save,res,sig,n,m,thetavar,sigma,t], f)
