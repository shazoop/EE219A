import torch
from NR import *


def solver(f,df,q,dq, init_x, u, funargs, NRparms, method = 'Trap', tstart = 0, tstop = 1, h = .05, record = True):
    '''
    f/df/q/qd should be functions of x,u, funargs
    funargs = [W,A]
    u is the current frame
    '''
    t = 0
    x_old = init_x
    
    if method == 'Trap':
        def gfun(x, x_old):
            retval = q(x, u, funargs) - q(x_old, u ,funargs) - h*(.5)*(f(x,u,funargs) + f(x_old,u,funargs))
            return retval

        def dgfun(x,x_old):
            retval = dq(x, u,funargs) - h*(.5)*(df(x,u,funargs))
            return retval
    elif method == 'BE':
        def gfun(x, x_old):
            retval = q(x, u, funargs) - q(x_old, u ,funargs) - h*f(x,u,funargs)
            return retval

        def dgfun(x,x_old):
            retval = dq(x, u,funargs) - h*(.5)*(df(x,u,funargs))
            return retval
    elif method == 'FE':
        def gfun(x, x_old):
            retval = q(x, u, funargs) - q(x_old, u ,funargs) - h*f(x_old,u,funargs)
            return retval

        def dgfun(x,x_old):
            retval = dq(x, u,funargs)
            return retval
    else:
        print('Unrecongnized or empty simulation method specified')
        return()
        
    
    
    if record == True:
        tpts = [0]
        xpts = init_x.unsqueeze(dim = 0)
        step = 0
        while (t < tstop):
            x_new, iters, success = NR(gfun,dgfun,x_old, x_old, NRparms)
            x_old = x_new
            xpts = torch.cat((xpts,x_new.unsqueeze(dim = 0)), dim = 0)
            tpts.append(t)
            if success != 1:
                print('NR failed to converge at time %s, step %s' % (t,step))
                return (tpts,xpts)
            t = t+h            
            step = step + 1
        return (tpts, xpts)
    else:
        while (t < tstop):
            x_new, iters, success = NR(gfun,dgfun,x_old, x_old, NRparms)
            x_old = x_new
            if success != 1:
                print('NR failed to converge at time %s, step %s' % (t,step))
                return x_new
            t = t+h            
            step = step + 1
        return x_new  


class DAE_solver:
    def __init__(self, DAE, init_x, u, funargs, method = 'Trap', NRparms =\
                 {'maxiter': 100, 'reltol': 1e-6, 'abstol': 1e-9, 'restol': 1e-12, 'convcrit': 'dxORres' }, tstart = 0, tstop = 1, h = .05,\
                 record = True):
        self.NRparms = NRparms
        self.DAE = DAE #the DAE class
        self.init_x = init_x #intial guess
        self.u = u #should be current movie frame
        self.funargs = funargs #should be [W,A], list of the weight matrices
        self.method = method #default to trap
        self.h = h #time step. default to .05
        self.tstop = tstop
        self.tstart = tstart
        self.record = record
        
    
    def getfuns(self):
        
        f = lambda x,u,funargs: self.DAE.f(x,u,funargs)
        q = lambda x,u,funargs: self.DAE.q(x,u,funargs)
        df = lambda x,u,funargs: self.DAE.df_dx(x,u,funargs)
        dq = lambda x,u,funargs: self.DAE.dq_dx(x,u,funargs)
        
        return (f,q,df,dq)
    
    def solve(self):
        f,q,df,dq = self.getfuns()
        if self.record == True:
            tpts, xpts = solver(f,df,q,dq, self.init_x, self.u, self.funargs, self.NRparms, self.method,\
                                self.tstart, self.tstop, self.h, self.record)
            return (tpts,xpts)
        else:
            xfinal = solver(f,df,q,dq, self.init_x, self.u, self.funargs, self.NRparms, self.method,\
                                self.tstart, self.tstop, self.h, self.record)
            return xfinal