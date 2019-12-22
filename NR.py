import torch


def NR(gfun,dgdxfun,init_x, fargs = [], NRparms = {'maxiter': 100, 'reltol': 1e-6, 'abstol': 1e-9, 'restol': 1e-12 , 'convcrit':'dxORres'}, detail = 0):
    '''
    g,dg_dx should both be functions of x,fargs
    No init/limiting used
    '''
    maxiter, reltol, abstol, restol, convcrit = NRparms.values()
    x = init_x
    tol = reltol*torch.norm(x) + abstol
    dx = 2*tol
    g = 2*tol
    def converged(g,dx, dxtol, residualtol, convcrit):
        if convcrit == 'dx':
            return(torch.norm(dx)<=dxtol);
        elif convcrit == 'res':
            return(torch.norm(g)<=residualtol);
        elif convcrit == 'dxORres':
            return( (torch.norm(dx)<=dxtol) or (torch.norm(g)<=residualtol) );
        elif convcrit == 'dxANDres':
            return( (torch.norm(dx)<=dxtol) and (torch.norm(g)<=residualtol) );
        else:
            print('unknown convergence criterion: %s, aborting.' \
                                                        % convcrit);
            return(-1)
    itr = 0
    while ((1 != converged(g,dx,tol,restol, convcrit)) and (itr < maxiter)):
        g = gfun(x, fargs)
        Jg = dgdxfun(x, fargs)
        dx = -torch.inverse(Jg) @ g
        x = x + dx
        itr = itr + 1
        
    if (itr == maxiter):
        solution = x
        iters = maxiter
        print('\nNR failed to solve nonlinear equations - reached maxiter=%d' % maxiter);
        success = 0
    elif (torch.sum(torch.isnan(x)) > 0):
        success = -1;
        print('\nNR appeared to complete, but contains NaN entries.');
    else:
        success = 1
                
    if detail == 1:
        print('\nNR succeeded in %d iterations.' % itr);

    solution = x
    iters = itr
    return((solution, iters, success))
