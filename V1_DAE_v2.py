import numpy as np
import torch

def sigmoid(x):
    return 1/(1+torch.exp(-x.clamp(min=-50,max=50)))

def dsig(x):
    return sigmoid(x)*(1-sigmoid(x))

def smrelu(x):
    return (1+torch.exp(x.clamp(min=-50,max=50)))

def dsmrelu(x):
    return sigmoid(x)

def logG(x):
    return .5*torch.log((1+x**2))

def dlogG(x):
    return x/(1+x**2)

def drelu(x):
    return .5*(torch.sign(x)+1)

class fullV1:
    def __init__(self, pixDim = 28, n1 = 12, d1 = 16, d2 = 16, S1 = 4, lam = .5, a = .7, b= .8, tau = 12.5):
        self._parms = {'pixel dim' : pixDim, 'RF 1 dim': n1, 'hidden layer depth': d1,\
                       'top layer depth' : d2, 'L1 stride': S1,'self-activate  parameter' : lam,  'a' : a, 'b': b, 'tau': tau} 
    
    def summary(self):
        '''All dimensions are one-sided. ex. if N = 4, then input image is N x N array
        N refers to pixel dim
        n refers to RF dim
        d refers to depth. ie. how many neurons have overlapping RFs
        greek letters are time constants
        S is stride size, how much displacement RFs are
        h is FE time step'''
        N,n1,d1,d2,S1,lam, a, b, tau = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = int(L1sq*d1)
        N2 = d2
        ttlN = 4*N1 + 2*N2

        out = ['In the hidden layer, there are %s units per slice, with depth %s, and RF size: %s by %s' % (L1_dim**2,d1,n1,n1)]
        out.append('In the top layer, there are %s units' % d2)
        out.append('There are %s hidden units, %s top units, and %s units in total.' % (N1,N2, ttlN))
        out.append('Fitz-Nag parameters (a,b, tau) are (%s,%s, %s).' % (a,b, tau))
        out.append('Self-activation parameter is %s' % lam)
        return out

    def print_allparms(self):
        return(self._parms)
    
    def print_parms(self):
        N,n1,d1,d2,S1,lam, a, b, tau = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        return N, L1_dim, d1,d2
    
    def pixel_mask(self, u):
        N,n1,d1,d2,S1,lam, a, b, tau = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        retval = torch.empty(L1_dim,L1_dim, n1**2)
        for i in range(L1_dim):
            for j in range(L1_dim):
                retval[i,j,:] = torch.flatten(u[(S1*i):(n1 + S1*i),(S1*j):(n1 + S1*j)])
        return retval

#     def pixel_mask(self, u):
#         N,n1,d1,d2,S1,lam, a, b, tau = self._parms.values()
#         L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
#         L1sq = int(L1_dim**2)
#         n1sq = int(n1**2)
#         retval = torch.zeros(L1sq*n1sq)
        
#         #If originally in (a,b, x), where (a,b) are Cartesian coords, then now we flatten along a row, then go down. Start at top left.
#         for i in range(L1_dim):
#             for j in range(L1_dim):
#                 retval[i*L1sq*n1sq + j*n1sq, i*L1sq*n1sq + (j+1)*n1sq] = torch.flatten(u[(S1*i):(n1 + S1*i),(S1*j):(n1 + S1*j)])
#         return retval
                    
#     def layer_mask(self, Input,Outdim, RFdim,S):
#         '''Given input (a,b,c) array, where:
#         1. a,b are "Cartesian" coordinates
#         2. c is depth index
#         and Output is Outdim x Outdim in "Cartesian coordinates" (overlapping RFs),
#         want to return a (Outdim,Outdim, c*RFdim^2). That is, for every "Cartesian" coord, will return a vectorized
#         version of the input to each RF
#         '''
#         a,b,c = Input.shape
#         retval = torch.empty(Outdim,Outdim,c*RFdim**2)
#         for i in range(Outdim):
#             for j in range(Outdim):
#                 retval[i,j,:] = torch.flatten(Input[(S*i):(RFdim + S*i),(S*j):(RFdim + S*j),:])
#         return retval
            
    def init(self):
        N,n1,d1,d2,S1,lam, a, b, tau = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = int(L1sq*d1)
        N2 = d2
        ttlN = 4*N1 + 2*N2
        x = torch.zeros(ttlN)
        W = torch.rand(L1_dim,L1_dim,d1,int(n1**2))
        A = torch.rand(N2,N1)
        return (x,[W,A])
    
    def xinit(self):
        N,n1,d1,d2,S1,lam, a, b, tau = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = int(L1sq*d1)
        N2 = d2
        ttlN = 4*N1 + 2*N2
        x = torch.zeros(ttlN)
        return (x)

#     def testf(self,x,u):
#         N,n1,d1,n2,d2,n3,d3, alpha, beta, gamma, eta, theta, p, S1, S2, S3, h = self._parms.values()
#         y,W_y,Q,t,z,W_z,z_bar,v,W_v,v_bar = x
#         L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
#         L2_dim = int((L1_dim + S2 - n2)/S2)
#         L3_dim = int((L2_dim + S3 - n3)/(S3))
        
#         y_in = self.layer_mask(y, L2_dim, n2, S2)
#         z_in = self.layer_mask(z, L3_dim, n3, S3)
#         u_in = self.pixel_mask(u)
#         return list((y_in,z_in,u_in))
    def uMul(self,u_original,funargs):
        '''
        Funargs is [W,A]
        '''
        W, A = funargs
        u_in = self.pixel_mask(u_original)
        u = torch.flatten(torch.einsum('abcn,abn -> abc', W, u_in))
        
        return u
    
    def getVars(self,x):
        N,n1,d1,d2,S1, lam, a, b, tau = self._parms.values()
        itau = 1/tau
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = L1sq*d1
        N2 = d2
        ttlN = 4*N1 + 2*N2

        y = x[:N1]
        v = x[N1:2*N1]
        w = x[2*N1:3*N1]
        v_z = x[3*N1: (3*N1+N2)]
        w_z = x[(3*N1 + N2): (3*N1 + 2*N2)]
        z = x[(3*N1 + 2*N2):]
        
        return (y,v,w,v_z,w_z,z)

    
    def f(self,x,u,funargs):
        '''
        x is 1-D tensor with shape (3*N1 + 3*N2): N1 is the ttl number of units in hidden layer. N2 is ttl for top layer.
        x = [y (N1) | v (N1) | w (N1) | v_z (N2) | w_z (N2) | z (N1) ]
        Eqns:
        dy = relu(u - Q @ y) - y. W is weight matrix. Q is fixed inhib matrix
        dv = v - v^3/3 - w + y
        dw = 1/tau(v + a - bw)
        dv_z = v_z - v_z^3/3 - w_z + A @ G(z). A is (N2,N1) weight matrix. G is gain function.
        dw_z = 1/tau(v_z + a - bw_z)
        z = relu(G(v) - Q_z @ z) - z. competitive dynamics among the v's.
        
        u is 1-D tensor with shape (N1), equal to x.
        '''
        N,n1,d1,d2,S1, lam, a, b, tau = self._parms.values()
        itau = 1/tau
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = L1sq*d1
        N2 = d2
        ttlN = 4*N1 + 2*N2
        W,A = funargs
        
        y,v,w,v_z,w_z,z = self.getVars(x)
        
        Q = torch.zeros(N1,N1)
        Q2 = torch.ones(N1,N1) - (1+lam)*torch.eye(N1)
        Q1 = (torch.ones(d1,d1) - (1+lam)*torch.eye(d1))
        for i in range(L1sq):
            Q[i*d1:(i+1)*d1, i*d1:(i+1)*d1] = Q1
        
#         yval = smrelu(.1*u - Q @ y) - y
        yval = -((.5*logG(u) - Q @ y).clamp(min=0).sub(y))
#         yval = (sigmoid(logG(u) - Q @ y)).sub(y)
#         yval = (u - Q @ torch.abs(y)).clamp(min=0) - y
        vval = v - (1/3)*(v**3) - w + y
        wval = itau*(v + a - b*w)
        v_zval = v_z - (1/3)*(v_z**3) - w_z + A @ z
#         v_zval = v_z - (1/3)*(v_z**3) - w_z + A @ v
        w_zval = itau*(v_z + a - b*w_z)
#         z = (v_z - Q2 @ z).clamp(min=0).sub(z) 
#         z = -((logG(v) - Q2 @ z).clamp(min=0).sub(z))
        z = -((v.clamp(min=0) - Q2 @ z).clamp(min=0).sub(z))
#         z = (sigmoid(logG(v_z) - Q2 @ z)).sub(z) 
#         z = (v_z - Q2 @ torch.sign(z)).clamp(min=0).sub(z) 


#         Q = torch.zeros(N1,N1)
#         Q2 = torch.ones(N1,N1) - (1+lam)*torch.eye(N1)
#         Q1 = (torch.ones(d1,d1) - (1+lam)*torch.eye(d1))
#         for i in range(L1sq):
#             Q[i*d1:(i+1)*d1, i*d1:(i+1)*d1] = Q1
        
# #         yval = smrelu(.1*u - Q @ y) - y
#         yval = sigmoid(u - Q @ y).sub(y)
# #         yval = (sigmoid(logG(u) - Q @ y)).sub(y)
# #         yval = (u - Q @ torch.abs(y)).clamp(min=0) - y
#         vval = v - (1/3)*(v**3) - w + logG(y)
# #         vval = v - (1/3)*(v**3) - w + y
#         wval = itau*(v + a - b*w)
#         v_zval = v_z - (1/3)*(v_z**3) - w_z + A @ logG(z)
# #         v_zval = v_z - (1/3)*(v_z**3) - w_z + A @ v
#         w_zval = itau*(v_z + a - b*w_z)
# #         z = (v_z - Q2 @ z).clamp(min=0).sub(z) 
#         z = (logG(v) - Q2 @ z).clamp(min=0).sub(z) 
# #         z = (sigmoid(logG(v_z) - Q2 @ z)).sub(z) 
# #         z = (v_z - Q2 @ torch.sign(z)).clamp(min=0).sub(z) 

        return torch.cat((yval,vval,wval,v_zval,w_zval,z))
    
    def q(self,x,u, funargs):
        return -x
    
    def df_dx(self,x,u,funargs):
        N,n1,d1,d2,S1, lam, a, b, tau = self._parms.values()
        itau = 1/tau
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = L1sq*d1
        N2 = d2
        ttlN = 4*N1 + 2*N2
        y,v,w,v_z,w_z,z = self.getVars(x)
        W,A = funargs
        
        J = torch.zeros(ttlN,ttlN)   
        #Q. dy/dy
        Q = torch.zeros(N1,N1)
        Q1 = (torch.ones(d1,d1) - (1+lam)*torch.eye(d1))
        for i in range(L1sq):
            Q[i*d1:(i+1)*d1, i*d1:(i+1)*d1] = Q1
        J[:N1,:N1] = -(torch.diag(drelu(.5*logG(u) - Q @ y)) @ (-Q) - torch.eye(N1)) 
#         J[:N1,:N1] = torch.diag(dsmrelu(u - Q @ y)) @ Q @torch.diag(torch.sign(y)) - torch.eye(N1)    
        #dv/dy
        J[N1:2*N1,:N1] = torch.eye(N1)
#         J[N1:2*N1,:N1] = torch.diag(dlogG(y))
        #dv/dv
        J[N1:2*N1,N1:2*N1] = torch.diag(1-v**2)
        #dv/dw
        J[N1:2*N1,2*N1:3*N1] = -torch.eye(N1)
        
        #dw/dv
        J[2*N1:3*N1,N1:2*N1] = itau*torch.eye(N1)
        J[2*N1:3*N1,2*N1:3*N1] = -b*itau*torch.eye(N1)
        
        #dv_z/dz
#         J[3*N1:(3*N1 + N2),N1:2*N1] = A @ torch.diag(dlogG(z))
        J[3*N1:(3*N1 + N2),(3*N1+2*N2):] = A

        #dv_z/dv_z
        J[3*N1:(3*N1 + N2),3*N1:(3*N1 + N2)] = torch.diag(1-v_z**2)
        #dv_z/dw_z
        J[3*N1:(3*N1 + N2),(3*N1 + N2):(3*N1 + 2*N2)] = -torch.eye(N2)
        
        #dw_z/dv_z
        J[(3*N1 + N2):(3*N1 + 2*N2),3*N1:(3*N1 + N2)] = itau*torch.eye(N2)
        #dw_z/dw_z
        J[(3*N1 + N2):(3*N1 + 2*N2),(3*N1 + N2):(3*N1 + 2*N2)] = -b*itau*torch.eye(N2)
        
        Q2 = torch.ones(N1,N1) - (1+lam)*torch.eye(N1)
        #dz/dv
#         J[(3*N1 + 2*N2):,N1:2*N1] = -torch.diag(drelu(logG(v) - Q2 @ z)) @ torch.diag(dlogG(v))
        J[(3*N1 + 2*N2):,N1:2*N1] = -torch.diag(drelu((v.clamp(min=0)) - Q2 @ z)) @ torch.diag(drelu(v))
#         J[(3*N1 + 2*N2):,3*N1:(3*N1 + N2)] = torch.diag(.5*(torch.sign(v_z - Q2 @ z)+1))
        #dz/dz
#         J[(3*N1 + 2*N2):, (3*N1 + 2*N2):] = -((torch.diag(drelu(logG(v) - Q2 @ z)) @ (-Q2)) - torch.eye(N1))
        J[(3*N1 + 2*N2):, (3*N1 + 2*N2):] = -((torch.diag(drelu((v.clamp(min=0)) - Q2 @ z)) @ (-Q2)) - torch.eye(N1)
        
    
#         J = torch.zeros(ttlN,ttlN)   
#         #Q. dy/dy
#         Q = torch.zeros(N1,N1)
#         Q1 = (torch.ones(d1,d1) - (1+lam)*torch.eye(d1))
#         for i in range(L1sq):
#             Q[i*d1:(i+1)*d1, i*d1:(i+1)*d1] = Q1
#         J[:N1,:N1] = torch.diag(drelu(logG(u) - Q @ y)) @ (-Q) - torch.eye(N1)  
# #         J[:N1,:N1] = torch.diag(dsmrelu(u - Q @ y)) @ Q @torch.diag(torch.sign(y)) - torch.eye(N1)    
#         #dv/dy
#         J[N1:2*N1,:N1] = torch.eye(N1)
# #         J[N1:2*N1,:N1] = torch.diag(y)
#         #dv/dv
#         J[N1:2*N1,N1:2*N1] = torch.diag(1-v**2)
#         #dv/dw
#         J[N1:2*N1,2*N1:3*N1] = -torch.eye(N1)
        
#         #dw/dv
#         J[2*N1:3*N1,N1:2*N1] = itau*torch.eye(N1)
#         J[2*N1:3*N1,2*N1:3*N1] = -b*itau*torch.eye(N1)
        
#         #dv_z/dz
#         J[3*N1:(3*N1 + N2),N1:2*N1] = A @ torch.diag(dlogG(v))
# #         J[3*N1:(3*N1 + N2),(3*N1+2*N2):] = A

#         #dv_z/dv_z
#         J[3*N1:(3*N1 + N2),3*N1:(3*N1 + N2)] = torch.diag(1-v_z**2)
#         #dv_z/dw_z
#         J[3*N1:(3*N1 + N2),(3*N1 + N2):(3*N1 + 2*N2)] = -torch.eye(N2)
        
#         #dw_z/dv_z
#         J[(3*N1 + N2):(3*N1 + 2*N2),3*N1:(3*N1 + N2)] = itau*torch.eye(N2)
#         #dw_z/dw_z
#         J[(3*N1 + N2):(3*N1 + 2*N2),(3*N1 + N2):(3*N1 + 2*N2)] = -b*itau*torch.eye(N2)
        
#         Q2 = torch.ones(N1,N1) - (1+lam)*torch.eye(N1)
#         #dz/dv
#         J[(3*N1 + 2*N2):,N1:2*N1] = torch.diag(drelu(logG(v) - Q2 @ z)) @ torch.diag(dlogG(v))
# #         J[(3*N1 + 2*N2):,N1:2*N1] = torch.diag(drelu((v.clamp(min=0)) - Q2 @ z)) @ torch.diag(drelu(v))
# #         J[(3*N1 + 2*N2):,3*N1:(3*N1 + N2)] = torch.diag(.5*(torch.sign(v_z - Q2 @ z)+1))
#         #dz/dz
#         J[(3*N1 + 2*N2):, (3*N1 + 2*N2):] = (torch.diag(drelu(logG(v) - Q2 @ z)) @ (-Q2)) - torch.eye(N1)
# #         J[(3*N1 + 2*N2):, (3*N1 + 2*N2):] = (torch.diag(drelu((v.clamp(min=0)) - Q2 @ z)) @ (-Q2)) - torch.eye(N1)

        return J
    
    def dq_dx(self,x,u, funargs):
        N,n1,d1,d2,S1, lam, a, b, tau = self._parms.values()
        L1_dim = int((N+S1-n1)/S1) #one-sided: number of units
        L1sq = int(L1_dim**2)
        N1 = L1sq*d1
        N2 = d2
        ttlN = 4*N1 + 2*N2
        
        J = torch.eye(ttlN)
        
        return -J

        
        
        
            
        

        
