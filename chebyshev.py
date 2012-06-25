from numpy import arange, outer, sin, cos, pi, arccos, dot, linspace, exp
from scipy.linalg import solve
import sympy, sympy.abc
from pylab import zeros, plot, legend, grid, show

a = 0.0 # lower integration limit
b = 1.0 # upper integration limit
N = 20 # number of points on grid


# test function
def testFunc(x):
    return sin(x)

# transform [-1,1] to [a,b]
def affineTransform(x, a, b):
    return (a+b)/2.0 + ((b-a)/2.0)*x

# integrate function analytically in [-1,1]
def integrate(C, a, b):
    I = 0.0
    c = (b-a)/2.0
    for i in range(0, N):
        I += c*C[i]*sympy.integrate(sympy.cos(i*sympy.acos(sympy.abc.x))/(pow(2.0, i-1)), (sympy.abc.x, -1.0, 1.0))
    print("polynomial approximation, I = " + str(I))
    I_analytical = sympy.integrate(sympy.sin(sympy.abc.x), (sympy.abc.x, a, b))
    print("analytical solution, I_analytical = " + str(I_analytical))

# Form a Chebyshev mesh and perform the Chebyshev interpolation
# N points on the mesh
def chebInterpol(N, a, b):
    i = arange(N)
    # Chebyshev nodes on [-1,1]
    x = -cos(pi*(2.0*i + 1.0)/(2.0*(N-1) + 2.0)) 
    
    # form the (normalized) Chebyshev-Vandermonde matrix
    TNx = cos(outer(arccos(x), i))/(pow(2.0, i-1))    
    # scale to physical interval [a,b]
    x = affineTransform(x, a, b)    

    f = testFunc(x)
    # C holds the coefficient of each Cheb harmonic
    C = solve(TNx, f)

    # integrate the interpolation function over [a,b]
    integrate(C, a, b)
    
    # form a uniform grid on [-1,1] to test interpolation
    xnew = linspace(-1.0, 1.0, N)
    # form the C-V matrix for new grid
    TNxnew = cos(outer(arccos(xnew), i))/(pow(2.0, i-1))
    
    # scale to [a,b]
    xnew = affineTransform(xnew, a, b)
    fnew = dot(TNxnew, C)
    
    # plot the Chebyshev nodes and the interpolation
    plot(x, f, 'r.', label = "func(Chebyshev nodes)")
    plot(xnew, fnew, 'g.', label = "Chebyshev interpolation")
    legend(loc="best")
    grid(True)
    show()

chebInterpol(N, a, b)


