import numpy as np
import torch

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def smrelu(x):
    return (1/3)*torch.log(1+torch.exp(3*x))

def relu(x):
    return max(0,x)

class FE_V1:
    def __init__(self, pixDim = 28, n1 = 12, d1 = 16, n2 = 3, d2 = 16, n3 = 2, d3 = 32, alpha = 1e-2, beta = 1e-6, gamma = 1e-6,\
                 eta = 2e-6, theta = 3e-6, lam = 1, p = 1/8, S1 = 4, S2 = 1, S3 = 1, h = .05):
        self._parms = {'pixel dim' : pixDim, 'RF 1 dim': n1, 'layer1 depth': d1,\
                       'RF 2 dim': n2, 'layer 2 depth': d2, 'RF 3 dim': n3,'layer 3 depth': d3,\
                      'L1Hebb': beta, 'L1AntiH': alpha, 'L1thresh': gamma, 'L2Hebb': eta, 'L3Hebb': theta,\
                       'L1prob': p, 'L1 stride': S1, 'L2 stride': S2, 'L3 stride': S3, 'time step': h} 
        
    def summary(self):
        '''All dimensions are one-sided. ex. if N = 4, then input image is N x N array
        N refers to pixel dim
        n refers to RF dim
        d refers to depth. ie. how many neurons have overlapping RFs
        greek letters are time constants
        S is stride size, how much displacement RFs are
        h is FE time step'''
        N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3,h = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L2_dim = int((L1_dim + S2 - n2)/S2)
        L3_dim = int((L2_dim + S3 - n3)/(S3))
        hold = (L1_dim,L2_dim,L3_dim)
        hold2 = (n1,n2,n3)
        hold3 = (d1,d2,d3)
        out = []
        for i in range(3):
            out.append('Layer %s has %s units per layer, depth %s, and RF size: %s by %s' % (i+1, hold[i]**2, hold3[i], hold2[i], hold2[i]))
        out.append('Time step: %s s' % h)
        return out
    
    def print_parms(self):
        return(self._parms)
        
    def pixel_mask(self, u):
        N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3, h = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        retval = torch.empty(L1_dim,L1_dim, n1**2)
        for i in range(L1_dim):
            for j in range(L1_dim):
                retval[i,j,:] = torch.flatten(u[(S1*i):(n1 + S1*i),(S1*j):(n1 + S1*j)])
        return retval
                    
    def layer_mask(self, Input,Outdim, RFdim,S):
        '''Given input (a,b,c) array, where:
        1. a,b are "Cartesian" coordinates
        2. c is depth index
        and Output is Outdim x Outdim in "Cartesian coordinates" (overlapping RFs),
        want to return a (Outdim,Outdim, c*RFdim^2). That is, for every "Cartesian" coord, will return a vectorized
        version of the input to each RF
        '''
        a,b,c = Input.shape
        retval = torch.empty(Outdim,Outdim,c*RFdim**2)
        for i in range(Outdim):
            for j in range(Outdim):
                retval[i,j,:] = torch.flatten(Input[(S*i):(RFdim + S*i),(S*j):(RFdim + S*j),:])
        return retval
            
    def x_init(self, W_y = None, Q = None, t= None, W_z = None, W_v = None):
        N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3, h = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L2_dim = int((L1_dim + S2 - n2)/S2)
        L3_dim = int((L2_dim + S3 - n3)/(S3))
        
        #Initialize all values:
        #Layer 1
        y = torch.zeros(L1_dim, L1_dim,d1) #cart coord, depth
        if W_y is None:
            W_y = torch.rand(L1_dim, L1_dim,d1, n1**2) #cart coord, depth, size of RF
        if Q is None:
            Q = torch.zeros(L1_dim, L1_dim, d1, d1) #cart coord, then inhibitory. anti-Hebb matrix
        if t is None:
            t = torch.zeros(L1_dim,L1_dim, d1)
        #Layer 2
        z = torch.zeros(L2_dim,L2_dim,d2)
        if W_z is None:
            W_z = torch.rand(L2_dim,L2_dim,d2, d1*n2**2)
        z_bar = torch.zeros_like(z)
        #Layer 3
        v = torch.zeros(L3_dim,L3_dim,d3)
        if W_v is None:
            W_v = torch.rand(L3_dim,L3_dim,d3, d2*n3**2)
        v_bar = torch.zeros_like(v)
        
        return list((y,W_y,Q,t,z,W_z,z_bar,v,W_v,v_bar))
    
    def testf(self,x,u):
        N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3, h = self._parms.values()
        y,W_y,Q,t,z,W_z,z_bar,v,W_v,v_bar = x
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L2_dim = int((L1_dim + S2 - n2)/S2)
        L3_dim = int((L2_dim + S3 - n3)/(S3))
        
        y_in = self.layer_mask(y, L2_dim, n2, S2)
        z_in = self.layer_mask(z, L3_dim, n3, S3)
        u_in = self.pixel_mask(u)
        return list((y_in,z_in,u_in))

    def f(self,x,u):
        N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3, h = self._parms.values()
        y,W_y,Q,t,z,W_z,z_bar,v,W_v,v_bar = x
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L2_dim = int((L1_dim + S2 - n2)/S2)
        L3_dim = int((L2_dim + S3 - n3)/(S3))
        z_inhib1 = torch.eye(d2).repeat(L2_dim,L2_dim,1,1)
        v_inhib1 = torch.eye(d3).repeat(L3_dim,L3_dim,1,1)
        y_inhib1 = torch.eye(d1).repeat(L1_dim,L1_dim,1,1)
        v_inhib = torch.ones_like(v_inhib1) - 1.5*v_inhib1
        z_inhib = torch.ones_like(z_inhib1) - 1.5*z_inhib1
        y_inhib = torch.ones_like(y_inhib1) - 1.5*y_inhib1
#         y_inhib = torch.einsum('abcd,abed -> abce', W_y, W_y)
#         for i in range(d1):
#             y_inhib[:,:,i,i] = 0
        
        
        y_in = self.layer_mask(y, L2_dim, n2, S2)
        z_in = self.layer_mask(z, L3_dim, n3, S3)
        u_in = self.pixel_mask(u)
        
        y_new = ((sigmoid(torch.einsum('abcn,abn -> abc',W_y,u_in)).sub(torch.einsum('abcd,abd -> abc',y_inhib,y))).clamp(min = 0).sub(y))
#         y_new = sigmoid(torch.einsum('abcn,abn -> abc',W_y,u_in).sub(torch.einsum('abcd,abd -> abc',y_inhib, y))).clamp(min= 0).sub(y)     
#         y_new = sigmoid(torch.einsum('abcn,abn -> abc',W_y,u_in).sub(torch.einsum('ab, abcd,abd -> abc', torch.norm(u_in,p = 1, dim=-1)**2,y_inhib, y))).clamp(min=0).sub(y)
#         y_new = sigmoid(torch.einsum('abcn,abn -> abc',W_y,u_in).add(torch.einsum('abcd,abd -> abc', Q, y)).sub(t)).sub(y)
        #y_new = 2*(smrelu(sigmoid(torch.einsum('abcn,abn -> abc',W_y,u_in)).sub(torch.einsum('abcd,abd -> abc',y_inhib,y))).sub(y))
        W_y_new = beta*(torch.einsum('abc,abn -> abcn',y,u_in).sub(torch.einsum('abc,abcn -> abcn',y,W_y)))
        #t_new = gamma*(y - p)
        t_new =t
        #Q_new = -alpha*(torch.einsum('abc,abd -> abcd',y,y) - p**2)
        Q_new = Q
        z_new = ((sigmoid(torch.einsum('abcn,abn -> abc',W_z,y_in)).sub(torch.einsum('abcd,abd -> abc',z_inhib,z))).clamp(min = 0).sub(z))
        W_z_new = eta*(torch.einsum('abc,abn -> abcn',z_bar,y_in).sub(torch.einsum('abc,abcn -> abcn',z_bar,W_z)))
        z_bar_new = (.5/h)*(z - z_bar)
        v_new = ((sigmoid(torch.einsum('abcn,abn -> abc',W_v,z_in)).sub(torch.einsum('abcd,abd -> abc',v_inhib,v))).clamp(min = 0).sub(v))
        W_v_new = theta*(torch.einsum('abc,abn -> abcn',v_bar,z_in).sub(torch.einsum('abc,abcn -> abcn',v_bar,W_v)))
        v_bar_new = (.5/h)*(v - v_bar)
        
        return list((y_new, W_y_new, Q_new, t_new, z_new,W_z_new,z_bar_new,v_new,W_v_new,v_bar_new))

def visZ(DAE, x, a,b,ix):
    '''Visualize higher layer weight matrices. For each "point" in the RF of a neuron, will visualize the lower layer's RF's.
    (a,b) are the Cartesian coordinates, while ix denotes the depth
    '''
    N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3, h = DAE.print_parms().values()
    L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
    L2_dim = int((L1_dim + S2 - n2)/S2)
    y,W_y,Q,t,z,W_z,z_bar,v,W_v,v_bar = x
    
    fig = plt.figure(figsize = (20,20))
    WZ = W_z[a,b,ix].reshape(n2,n2,d1)
    for i in range(n2):
        for j in range(n2):
            max_ix = torch.argmax(WZ[i,j,:])
            fig.add_subplot(n2,n2,n2*i + j + 1)
            plt.imshow(W_y[(S2*a+i), (S2*b+j), ix,:].reshape(n1,n1).cpu().numpy(), cmap = 'Greys')
            
    return 