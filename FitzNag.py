import torch

class FN:
    def __init__(self, a = .7, b= .8, tau = 12.5, n = None):
        if n is None:
            print('Error! Specify the number of neurons')
        self._parms = {'# neurons': n, 'a': a, 'b': b, 'tau': tau}
                       
    def print_parms(self):
        print(self._parms)                       
    
    def init(self):
        n,a,b,tau = self._parms.values()
        return torch.zeros(2*n)
                       
    def f(self,x,u, funargs = []):
        '''
        x should be tensor with shape (2n). First n entries are v. Latter half are w. (v[i], w[i+n]) are a pair.
        v, pulse, and second index is w, refrac. 
        u is tensor with shape (n). odd entries are 0
        
        '''
        n, a,b,tau = self._parms.values()
        v = x[:n]
        w = x[n:]   
        retval = torch.zeros_like(x)

        retval[:n] = v - (1/3)*(v**3) - w + u
        retval[n:] = v + a -b*w
        
        return -retval
    
    def q(self,x,u, funargs = []):
        n, a,b,tau = self._parms.values()
        v = x[:n]
        w = x[n:]   
        retval = torch.zeros_like(x)
        
        retval[:n] = -v
        retval[n:] = -tau*w
        
        return -retval
    
    def df_dx(self,x,u, funargs = []):
        n, a,b,tau = self._parms.values()
        v = x[:n]
        w = x[n:]   
        J = torch.zeros(2*n,2*n)

        J[:n,:n] = torch.diag(1-v**2) #dfv_dv
        J[:n,n:] = -1*torch.eye(n) #dfv_dw
        J[n:,:n] = torch.eye(n)
        J[n:,n:] = -b*torch.eye(n)
        
        return -J
    
    def dq_dx(self,x,u, funargs = []):
        n, a,b,tau = self._parms.values()
        v = x[:n]
        w = x[n:]   
        J = torch.zeros(2*n,2*n)
                       
        J[:n,:n] = -torch.eye(n)
        J[n:,n:] = -tau*torch.eye(n)
        
        return -J
    

        
       


        
        
        