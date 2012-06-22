from numpy import arange, outer, cos, pi, arccos, dot, linspace
from scipy.linalg import solve
from pylab import plot, legend, grid, show

a = 0.0
b = 1.0
N = 100 #number of points on grid

# test function
def testFunc(x):
    return cos(10*x)

# transform any interval to [a,b]
def affineTransform(x, a, b):
    return (a+b)/2.0 + ((b-a)/2.0)*x

# Form a Chebyshev mesh and perform the Chebyshev interpolation
# N points on the mesh
def chebInterpol(N, a, b):
    i = arange(N)
    # Chebyshev nodes on [-1,1]
    x = -cos(pi*(2.0*i + 1.0)/(2.0*(N-1) + 2.0)) 
    # form the (normalized) Chebyshev-Vandermonde matrix
    TNx = cos(outer(arccos(x), i))/(pow(2.0, N-1))
    
    # scale to physical interval [a,b]
    x = affineTransform(x, a, b)
    f = testFunc(x)
    C = solve(TNx, f)
    
    # form a uniform grid on [-1,1] check interpolation
    xnew = linspace(-1.0, 1.0, N)
    # form the C-V matrix for new grid
    TNxnew = cos(outer(arccos(xnew), i))/(pow(2.0, N-1))
    
    # scale to [a,b]
    xnew = affineTransform(xnew, a, b)
    fnew = dot(TNxnew, C)
    
    # plot the Chebyshev nodes and the interpolation
    plot(x, f, 'r.', label = "func(Chebyshev nodes)")
    plot(xnew, fnew, 'g-', label = "Chebyshev interpolation")
    legend(loc="best")
    grid(True)
    show()

chebInterpol(N, a, b)


