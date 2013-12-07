from numpy import arange, outer, sin, cos, pi, arccos, dot, linspace, exp
from scipy.linalg import solve
import sympy, sympy.abc
from pylab import zeros, plot, legend, grid, show

a = 0.0 # lower integration limit
b = 0.1 # upper integration limit
N = 10 # number of points on grid
beta = 10.0

# test function
def testFunc(x):
    return exp(-beta*x*x)

# transform [-1,1] to [a,b]
def affineTransform(x, a, b):
    return (a+b)/2.0 + ((b-a)/2.0)*x

# differentiation function in [a,b]
def differentiate(C,a, b):
    diff1 = [0]*N
    diff1_anal = [0]*N
    diff2 = [0]*N
    diff2_anal = [0]*N
    i = arange(N)
    # Chebyshev nodes on [-1,1]
    x = -cos(pi*(2.0*i + 1.0)/(2.0*(N-1) + 2.0)) 

    #chebdiff1 = pow(2,1-i)*i*sin(i*arccos(x))/pow(1-x*x,0.5)
    #chebdiff2 = pow(2,1-i)*i*( x*sin(i*arccos(x))/pow(1-x*x,1.5) - i*cos(i*arccos(x))/(1-x*x))

    # scale to physical interval [a,b]
    xp = affineTransform(x, a, b)
    
    c = 2.0/(b-a)
    for i in range(0,N):
        for j in range(0, N):
            tmp = pow(2,1-j)*j*sin(j*arccos(x[i]))/pow(1-x[i]*x[i],0.5)
            diff1[i] += c*C[j]*tmp
            tmp = pow(2,1-j)*j*( x[i]*sin(j*arccos(x[i]))/pow(1-x[i]*x[i],1.5) - j*cos(j*arccos(x[i]))/(1-x[i]*x[i]))
            diff2[i] += c*c*C[j]*tmp
        diff1_anal[i] = (sympy.diff(sympy.exp(-sympy.abc.x*sympy.abc.x*beta), sympy.abc.x, 1)).subs(sympy.abc.x,xp[i])
        diff2_anal[i] = (sympy.diff(sympy.exp(-sympy.abc.x*sympy.abc.x*beta), sympy.abc.x, 2)).subs(sympy.abc.x,xp[i])
        
    # plot the Chebyshev nodes and the interpolation
    plot(xp, diff1, 'r--', label = "chebyshev derivative ord 1")
    plot(xp, diff1_anal, 'g.', label = "analytic derivative ord 1")
    plot(xp, diff2, 'y--', label = "chebyshev derivative ord 2")
    plot(xp, diff2_anal, 'b*', label = "analytic derivative ord 2")
    legend(loc="best")
    grid(True)
    show()

# integrate function in [a,b]
def integrate(C, a, b):
    I = 0.0
    c = (b-a)/2.0
    for i in range(0, N):
        I += c*C[i]*sympy.integrate(sympy.cos(i*sympy.acos(sympy.abc.x))/(pow(2.0, i-1)), (sympy.abc.x, -1.0, 1.0))
    print("polynomial approximation, I = " + str(I))
    I_analytical = sympy.N(sympy.integrate(sympy.exp(-sympy.abc.x*sympy.abc.x*beta), (sympy.abc.x, a, b)))
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

    # differentiate the interpolation function over [a,b]
    differentiate(C,a,b)
    '''
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
    '''
chebInterpol(N, a, b)


