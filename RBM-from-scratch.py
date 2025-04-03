import numpy as np
import math

from time import time

class RBM():
    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hid = n_hid
        # Parameters
        self.W = 0.01 * np.random.randn(n_vis, n_hid)
        self.vbias = np.zeros(n_vis)
        self.hbias =np.zeros(n_hid)
        # Gradients
        self.W_grad = np.zeros(self.W.shape)
        self.vbias_grad = np.zeros(n_vis)
        self.hbias_grad = np.zeros(n_hid)
        # Velocities - for momentum
        self.W_vel = np.zeros(self.W.shape)
        self.vbias_vel = np.zeros(n_vis)
        self.hbias_vel = np.zeros(n_hid)
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    
    def h_given_v(self, v):
     
        p = self.sigmoid(np.matmul(v, self.W) + self.hbias)
        return (p, np.random.binomial(1, p=p))
    
    
    def v_given_h(self, h):
   
        p = self.sigmoid(np.matmul(h, self.W.T) + self.vbias)
        return (p, np.random.binomial(1, p=p))
    
    def compute_error_and_grads(self, batch, num_steps=1):

        b_size = batch.shape[0]
        v0 = batch.reshape(b_size, -1)
        
        # Compute gradients - Positive Phase
        ph0, h0 = self.h_given_v(v0)
        W_grad = np.matmul(v0.T, ph0)
        vbias_grad = np.sum(v0, axis=0)
        hbias_grad = np.sum(ph0, axis=0)
        
        # Compute gradients - Negative Phase

        pv1, v1 = self.v_given_h(h0)
        ph1, h1 = self.h_given_v(pv1)
        
        W_grad -= np.matmul(pv1.T, ph1)
        vbias_grad -= np.sum(pv1, axis=0)
        hbias_grad -= np.sum(ph1, axis=0)
        
        self.W_grad = W_grad/b_size
        self.hbias_grad = hbias_grad/b_size
        self.vbias_grad = vbias_grad/b_size
        
        recon_err = np.mean(np.sum((v0 - pv1)**2, axis=1), axis=0) # sum of squared error averaged over the batch
        return recon_err
    
    def update_params(self, lr, momentum=0):
    
        self.W_vel *= momentum
        self.W_vel += (1.-momentum) * lr * self.W_grad
        self.W += self.W_vel
        
        self.vbias_vel *= momentum
        self.vbias_vel += (1.-momentum) * lr * self.vbias_grad
        self.vbias += self.vbias_vel
        
        self.hbias_vel *= momentum
        self.hbias_vel += (1.-momentum) * lr * self.hbias_grad
        self.hbias += self.hbias_vel
        
    def reconstruct(self, v):
        ph0, h0 = self.h_given_v(v)
        pv1, v1 = self.v_given_h(ph0)
        return v1



batch_size = 50
num_epochs = 15000
lr = 0.0001
num_steps = 1


def get_batches(data, batch_size, shuffle):

    if(shuffle):
        np.random.shuffle(data)
    if(batch_size == -1):
        batch_size = len(data)
    num_batches = math.ceil(data.shape[0]/batch_size)
    for batch_num in range(num_batches):
        yield data[batch_num*batch_size:(batch_num+1)*batch_size]   
        
       
#Data
SC = np.load('/home/mahdis/coded_6lat_T=3.5_600k.npy')
X_train=np.array(SC)

# Our RBM object
rbm = RBM(n_vis=36 , n_hid=36)

# Training loop
    
errors = []
start_time = time()


for epoch in range(1, num_epochs+1):
    iteration = 0
    error = 0
    
    for batch in get_batches(X_train, batch_size,shuffle=True):
        iteration += 1
        
        # Compute gradients and errors
        error += rbm.compute_error_and_grads(batch, num_steps=num_steps)
        
        # Update parameters - use momentum as explained in Hinton's guide
       # if(epoch > 1500):
        #    rbm.update_params(lr, momentum=0.3)
        #else:
        #    rbm.update_params(lr, momentum=0.9)
        rbm.update_params(lr) #Without using momentum with a fixed learning rate

    print("epoch:{} \t error:{:.4f} \t training time:{:.2f} s".format(epoch, error, time()-start_time))
    errors.append(error)
    
end_time = time()
TIME = end_time - start_time
print(TIME)
print(min(errors))



x_test= np.random.randint(2, size=(580000,36))
num=100

for i in range(num):
    x_test=rbm.reconstruct(x_test)


np.save('RBM_6lat_T=3.5_600k',x_test)
