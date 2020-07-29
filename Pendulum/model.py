
"""
January 2020
Inspired and based on the tensorflow CVAE tutorial

Sebastian Kaltenbach
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np



class MODEL(tf.keras.Model):
  def __init__(self):
    super(MODEL, self).__init__()
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(29, 29, 75 , 1)),
          tf.keras.layers.Conv3D(
              filters=32, kernel_size=(3,3,2), strides=(2, 2,2), activation='relu'),
          tf.keras.layers.Conv3D(
              filters=64, kernel_size=2, strides=(1, 1,1), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(149 + 149),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(1,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
             filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="VALID",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
             filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )
    # This versions uses point-estimates for (hyper)parameters
    self.Theta = tf.Variable(np.zeros((101,1), dtype=np.float32), name="Theta",trainable=True) #Parameters of CG evolution law 
    self.Sigma = tf.Variable(-6.*tf.ones(shape=(1,1)), name="Sigma",trainable=False) #parameter for enforcement of residual constraint
    self.Sigma_H = tf.Variable(-8.*tf.ones(shape=(1,1)), name="Sigma_H",trainable=True) # Parameter for slowness prior
    self.ARD = tf.Variable(10.*np.ones((101,1), dtype=np.float32), name="Theta_var",trainable=False) # Parameter for ARD Prior


  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits



@tf.function
def evolution_lib(z):
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(1,51):
        library.append(tf.sin(z[:,0]*i))
        
    for i in range(1,51):
        library.append(tf.cos(z[:,0]*i))

    return tf.stack(library, axis=1)




def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(tf.math.minimum(-logvar,20.)) + logvar + log2pi),
      axis=raxis)

def normal_pdf(sample, mean, std, raxis=1):
  return tf.reduce_sum(tf.exp(-.5 * ((sample - mean) ** 2. /std**2))/(2*np.pi*std**2)**0.5, axis=raxis)


@tf.function
def compute_loss(model, x,scale):
  a=16
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit_tmp=[]
  for i in range(75):
    x_logit_tmp.append(model.decode(tf.reshape(z[:,i],[a,1])))
  x_logit=tf.stack(x_logit_tmp)
  x_logit=tf.transpose(x_logit, [1, 2, 3, 0, 4])
    
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz=[]
  for i in range(73):
      logpz.append(log_normal_pdf(0., (tf.reshape(z[:,2+i],[a,1])-tf.reshape(z[:,i+1],[a,1])-0.05*tf.reshape(z[:,i+75+1],[a,1])), model.Sigma))
      logpz.append(log_normal_pdf(0., (tf.reshape(z[:,76+i],[a,1])-tf.reshape(z[:,i+75],[a,1])-tf.matmul(evolution_lib(tf.reshape(z[:,i+1],[a,1])),model.Theta)), model.Sigma))
  logpz.append(log_normal_pdf(0., tf.reshape(z[:,1],[a,1])-tf.reshape(z[:,0],[a,1])-0.05*tf.reshape(z[:,75],[a,1]), model.Sigma))
  logpz.append(log_normal_pdf(0.,tf.reshape(z[:,75],[a,1])- tf.matmul(evolution_lib(tf.reshape(z[:,0],[a,1])),model.Theta), model.Sigma))
  for i in range(74):
      logpz.append(log_normal_pdf(tf.reshape(z[:,i+1],[a,1]), tf.reshape(z[:,i],[a,1]), model.Sigma_H))

  logpzstacked=tf.stack(logpz)
  logpz2 = tf.math.log(0.5*normal_pdf(tf.reshape(z[:,0],[a,1]), 1.5, 1.5)+0.5*normal_pdf(tf.reshape(z[:,0],[a,1]), -1.5, 1.5))
  logqz_x = log_normal_pdf(z, mean, logvar)


  reg_s= -model.Sigma_H*((1e-9)-1.)-(1e-9)*tf.exp(-model.Sigma_H)
  logptheta = log_normal_pdf(model.Theta, 0., tf.math.log(1./model.ARD))

  return -(scale*tf.reduce_sum(logpx_z)  + tf.reduce_sum(logpzstacked)+tf.reduce_sum(logpz2)+reg_s - tf.reduce_sum(logqz_x)+tf.reduce_sum(logptheta))

@tf.function
def compute_apply_gradients(model, x, optimizer,scale):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x,scale)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
