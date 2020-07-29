"""
January 2020

@author: Sebastian Kaltenbach
"""

import model as m
import tensorflow as tf
import numpy as np



train_images=np.load('Data_pendulum.npy')
train_images = train_images.reshape(train_images.shape[0], 29, 29, train_images.shape[3],1).astype('float32')

TRAIN_BUF = 16
BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

epochs = 10000
model = m.MODEL()
optimizer = tf.keras.optimizers.Adam(5e-4)

for epoch in range(1, epochs + 1):
  for train_x in train_dataset:
    m.compute_apply_gradients(model, train_x, optimizer,tf.constant(1.-0.95*np.exp(-epoch/1000),dtype=tf.float32))


  if epoch % 500 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in train_dataset:
      loss(m.compute_loss(model, test_x,tf.constant(1.0,dtype=tf.float32)))
    elbo = -loss.result()
    print('Epoch: {}, Training set ELBO: {} '.format(epoch,elbo))
    
    
    
#Gradually decrease the variance of the residual constraint to the final value    
model.Sigma.assign(tf.constant(-8.,shape=(1,1)))   
for epoch in range(1, epochs + 1):
  for train_x in train_dataset:
    m.compute_apply_gradients(model, train_x, optimizer,tf.constant(1.-0.95*np.exp(-epoch/1000),dtype=tf.float32))

  #Closed-form update law for ARD-Prior. Closed form updates for other parameters are also possible
  tmp=[]
  for i in range(101):
    tmp.append((1e-8+0.5)/(1e-8+0.5*(model.Theta.numpy()[i]**2)))
  model.ARD.assign(tf.stack(tmp))

  if epoch % 500 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in train_dataset:
      loss(m.compute_loss(model, test_x,tf.constant(1.0,dtype=tf.float32)))
    elbo = -loss.result()
    print('Epoch: {}, Training set ELBO: {}'.format(epoch,elbo))
 
   
model.Sigma.assign(tf.constant(-10.,shape=(1,1)))   
for epoch in range(1, epochs + 1):
  for train_x in train_dataset:
    m.compute_apply_gradients(model, train_x, optimizer,tf.constant(1.-0.95*np.exp(-epoch/1000),dtype=tf.float32))
  tmp=[]
  for i in range(101):
    tmp.append((1e-8+0.5)/(1e-8+0.5*(model.Theta.numpy()[i]**2)))
  model.ARD.assign(tf.stack(tmp))

  if epoch % 500 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in train_dataset:
      loss(m.compute_loss(model, test_x,tf.constant(1.0,dtype=tf.float32)))
    elbo = -loss.result()
    print('Epoch: {}, Training set ELBO: {}'.format(epoch,elbo))
    
model.Sigma.assign(tf.constant(-11.,shape=(1,1)))   
for epoch in range(1, epochs + 1):
  for train_x in train_dataset:
    m.compute_apply_gradients(model, train_x, optimizer,tf.constant(1.-0.95*np.exp(-epoch/1000),dtype=tf.float32))
  tmp=[]
  for i in range(101):
    tmp.append((1e-8+0.5)/(1e-8+0.5*(model.Theta.numpy()[i]**2)))
  model.ARD.assign(tf.stack(tmp))

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in train_dataset:
      loss(m.compute_loss(model, test_x,tf.constant(1.0,dtype=tf.float32)))
    elbo = -loss.result()
    print('Epoch: {}, Training set ELBO: {}'.format(epoch,elbo))
    
for epoch in range(1, 2*epochs + 1):
  for train_x in train_dataset:
    m.compute_apply_gradients(model, train_x, optimizer,tf.constant(1.,dtype=tf.float32))
  tmp=[]
  for i in range(101):
    tmp.append((1e-8+0.5)/(1e-8+0.5*(model.Theta.numpy()[i]**2)))
  model.ARD.assign(tf.stack(tmp))

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in train_dataset:
      loss(m.compute_loss(model, test_x,tf.constant(1.0,dtype=tf.float32)))
    elbo = -loss.result()
    print('Epoch: {}, Training set ELBO: {}'.format(epoch,elbo))
    
model.save_weights('pendulum_results', save_format='tf')#Save results

