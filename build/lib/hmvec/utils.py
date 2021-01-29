import numpy as np
from scipy.interpolate import interp1d

def interp(x,y,bounds_error=False,fill_value=0.,**kwargs):
    return interp1d(x,y,bounds_error=bounds_error,fill_value=fill_value,**kwargs)

def vectorized_bisection_search(x,inv_func,ybounds,monotonicity,rtol=1e-4,verbose=True,hang_check_num_iter=20):
    """
    You have a monotonic one-to-one relationship x <-> y
    You know the inverse function inv_func=x(y), 
    but you don't know y(x).
    Find y for a given x using a bisection search
    assuming y is bounded in ybounds=(yleft,yright)
    and with a relative tolerance on x of rtol.
    """
    assert monotonicity in ['increasing','decreasing']
    mtol = np.inf
    func = inv_func
    iyleft,iyright = ybounds
    yleft = x*0+iyleft
    yright = x*0+iyright
    i = 0
    warned = False
    while np.any(np.abs(mtol)>rtol):
        ynow = (yleft+yright)/2.
        xnow = func(ynow)
        mtol = (xnow-x)/x
        if monotonicity=='decreasing':
            yleft[mtol>0] = ynow[mtol>0]
            yright[mtol<=0] = ynow[mtol<=0]
        elif monotonicity=='increasing':
            yright[mtol>0] = ynow[mtol>0]
            yleft[mtol<=0] = ynow[mtol<=0]
        i += 1
        if (i>hang_check_num_iter) and not(warned):
            print("WARNING: Bisection search has done more than ", hang_check_num_iter,
                  " loops. Still searching...")
            warned = True
    if verbose: print("Bisection search converged in ", i, " iterations.")
    return ynow


def test_bisection_search():
    true_y_of_x = lambda x: x**2.
    x_of_y = lambda y: np.sqrt(y)
    xs = np.array([2.,4.,6.])
    eys = np.array([4.,16.,36.])
    d = vectorized_bisection_search(xs,x_of_y,(1,40),'increasing',rtol=1e-4,verbose=True)
    assert np.all(np.isclose(d,eys,rtol=1e-3))

#test_bisection_search()    
