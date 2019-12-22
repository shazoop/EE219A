import numpy as np

def FE_solve(V1class, h, t_curr, x_in, ufun, Reset = 0,W_y = None, Q = None, t = None, W_z = None,W_v = None):
    if Reset == 1:
        x = V1class.x_init(W_y=W_y, Q = Q, t=t, W_z=W_z,W_v=W_v)
    else:
        x = x_in
    M = len(x)
    x_new = V1class.f(x,ufun(t_curr))
    for i in range(M):
        x_new[i] = x[i] + h*x_new[i]
    return x_new
    
def FE_movie(V1class, mov, timePerFrame, W_y = None, Q = None, t = None, W_z = None,W_v = None):
    T = mov.size(0)
    ufunc = mov2u(mov)
    _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,h = V1class.print_parms().values()
    numSteps = int(timePerFrame*T/h) - 1
    t_old = 0
    x = V1class.x_init(W_y = W_y, Q = Q, t = t, W_z = W_z,W_v = W_v)
    Reset = 1
    for step in range(numSteps):
        t_new = (t_old + h)/timePerFrame
        x = FE_solve(V1class, h, t_new, x, ufunc, Reset, W_y, Q, t, W_z, W_v)
        y,W_y,Q,t,z,W_z,z_bar,v,W_v,v_bar = x     
        t_old = t_new
        Reset = 0
    return x

def mov2u(mov):
    '''Given a input movie (t,n,n), will return the i^th frame, where i is integer floor of t.
    Every frame is a n by n array.
    '''
    return lambda t: mov[int(np.floor(t)),:]


    