from scipy.interpolate import BSpline
from scipy.integrate import quad
import numpy as np
import math
import matplotlib.pyplot as plt
import torch

def bs(x, knots, boundary_knots, degree = 3, intercept = False):
    """ Generate the B-spline basis matrix for a polynomial spline.

    This function mimick the function bs in R package splines

    Args:
        x (Tensor): Values at which basis functions are evaludated.
        knots (Tensor): Internal breakpoints that define the spline.
        boundary_knots (Tensor): Boundary points 
        degree (int, optional): The degree of the piecewise polynomial. 
            The default is 3 for cubic splines.
        intercept (bool, optional): If True, an intercept is included 
            in the basis. Default is False.
    
    Returns:
        design_matrix (Tensor): A tensor of dimension (len(x), df), 
            where df = len(knots) + degree if intercept = False, 
            df = len(knots) + degree + 1 if intercept = True.
    """

    knots = knots.numpy()
    boundary_knots = boundary_knots.numpy()
    x = x.numpy()
    
    knots = np.concatenate([knots, boundary_knots])
    knots.sort()

    augmented_knots = np.concatenate([np.array([boundary_knots[0] for i in range(degree + 1)]),
                                      knots,
                                      np.array([boundary_knots[1] for i in range(degree + 1)])])
    num_of_basis = len(augmented_knots) - 2*(degree + 1) + degree + 1

    spl_list = []
    for i in range(num_of_basis):
        coeff = np.zeros(num_of_basis)
        coeff[i] = 1.0
        spl = BSpline(augmented_knots, coeff, degree, extrapolate = False)
        spl_list.append(spl)

    design_matrix = np.array([spl(x) for spl in spl_list]).T

    ## if the intercept is Fales, drop the first basis term, which is often
    ## referred as the "intercept". Note that np.sum(design_matrix, -1) = 1.
    ## see https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf
    if intercept is False:
        design_matrix = design_matrix[:, 1:]

    design_matrix = torch.from_numpy(design_matrix)

    return design_matrix

def pbs(x, knots, boundary_knots = torch.tensor([-math.pi, math.pi]), degree = 3, intercept = False):
    """ Compute the design matrix of a periodic B-spline. 

    This function mimick the pbs function in R package pbs.

    Args:
        x (Tensor): Values at which basis functions are evaludated.
        knots (Tensor): Internal breakpoints that define the spline.
        boundary_knots (Tensor): Boundary points 
        degree (int, optional): The degree of the piecewise polynomial. 
            The default is 3 for cubic splines.
        intercept (bool, optional): If True, an intercept is included 
            in the basis. Default is False.

    Returns:
        design_matrix (Tensor): A tensor of dimension (len(x), df), 
            where df = len(knots) if intercept = False, 
            df = len(knots) + 1 if intercept = True.
    """

    knots = knots.numpy()
    boundary_knots = boundary_knots.numpy()
    x = x.numpy()
    
    knots = np.concatenate([knots, boundary_knots])
    knots.sort()

    augmented_knots = np.copy(knots)
    for i in range(degree):
        augmented_knots = np.append(augmented_knots, knots[-1] + knots[i+1] - knots[0])
    for i in range(degree):
        augmented_knots = np.insert(augmented_knots, 0, knots[0] - (knots[-1] - knots[-1-(i+1)]))

    num_of_basis = len(augmented_knots) - 2*(degree + 1) + degree + 1

    spl_list = []
    for i in range(num_of_basis):
        coeff = np.zeros(num_of_basis)
        coeff[i] = 1.0
        spl = BSpline(augmented_knots, coeff, degree, extrapolate = False)
        spl_list.append(spl)

    design_matrix = np.array([spl(x) for spl in spl_list]).T
    design_matrix_left = design_matrix[:, 0:degree]
    design_matrix_right = design_matrix[:, -degree:]
    design_matrix_middle = design_matrix[:, degree:-degree]
    design_matrix = np.concatenate([design_matrix_middle, design_matrix_left + design_matrix_right], axis = -1)

    ## if the intercept is Fales, drop the first basis term, which is often
    ## referred as the "intercept".
    ## see https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf
    if intercept is False:
        design_matrix = design_matrix[:, 1:]

    design_matrix = torch.from_numpy(design_matrix)
    
    return design_matrix

def bs_lj(r, r_min, r_max, num_of_basis, omega = False):
    ''' Compute the design matrix of a custimized B-spline 
    for Lennard-Jones type interaction.

    Args:
        r (Tensor): Distances at which basis functions are evaluated.
        r_min (float): A cutoff distance. 
            When r < r_min, the interaction becomes repulsive and 
            the basis function will provide a postive value. 
        r_max (float): A cutoff distance. 
            When r > r_max, all basis functions are zeros.
        num_of_basis (int): The number of basis.
        omega (bool): Integral of secondary derivatives.
            If True, the function will also return a matrix omega.
            Omega[i,j] = \int_{r_min}^{r_max} 
                         basis_i.derivative(2)*basis_j.derivative(2) dr. 
            This matrix is useful when fitting a smoothing splines
            by addding a penaly term controling the secondary 
            derivative of splines.

    Returns:
        design_matrix (Tensor): A matrix of dimension (len(r), num_of_basis).
        omega (Tensor): A matrix containing the integral of the splines' 
            second derivatives
    '''
    r = r.numpy()
    
    ## degree of spline    
    degree = 3

    ## knots of cubic spline
    t = np.linspace(r_min, r_max + (r_max - r_min), num_of_basis*2 + 3)

    ## number of basis
    n = len(t) - 2 + degree + 1

    ## preappend and append knots
    t = np.concatenate(
        (np.array([r_min for i in range(degree)]),
         t,
         np.array([r_max + (r_max - r_min) for i in range(degree)])
        ))

    spl_list = []
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        spl_list.append(BSpline(t, c, degree, extrapolate = True))

    spl_list = spl_list[:-(n//2+2)]    
    spl_list = [spl_list[i] for i in range(len(spl_list)) if i != 1]    

    design_matrix = []
    for i in range(len(spl_list)):
        u = spl_list[i](r)
        if i != 0:
            u[r <= r_min] = 0.
        design_matrix.append(u)
    design_matrix = np.array(design_matrix).T

    if omega:
        omega = np.zeros((len(spl_list), len(spl_list)))
        for i in range(len(spl_list)):
            for j in range(i, len(spl_list)):
                spl_i = spl_list[i].derivative(2)
                spl_j = spl_list[j].derivative(2)
                omega[i,j] = quad(lambda x: spl_i(x)*spl_j(x), r_min, r_max, limit = 10_000)[0]
                omega[j,i] = omega[i,j]
                
        omega[0,:] = 0.0
        omega[:,0] = 0.0
        
        return torch.from_numpy(design_matrix), torch.from_numpy(omega)

    else:
        return torch.from_numpy(design_matrix)

def bs_rmsd(r, r_max, num_of_basis):
    ''' Compute the design matrix of a custimized B-spline 
    for a biasing potential on RMSD.

    Args:
        r (Tensor): Distances at which basis functions are evaluated.
        r_max (float): A cutoff distance. 
            When r > r_max, all basis functions are zeros.
        num_of_basis (int): The number of basis.
    Returns:
        design_matrix (Tensor): A matrix of dimension (len(r), num_of_basis).
    '''
    r = r.numpy()
    
    ## degree of spline    
    degree = 3

    ## knots of cubic spline
    r_min = 0.0
    t = np.linspace(r_min, r_max + (r_max - r_min), num_of_basis*2 + 2)

    ## number of basis
    n = len(t) - 2 + degree + 1

    ## preappend and append knots
    t = np.concatenate(
        (np.array([r_min for i in range(degree)]),
         t,
         np.array([r_max + (r_max - r_min) for i in range(degree)])
        ))

    spl_list = []
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        spl_list.append(BSpline(t, c, degree, extrapolate = True))

    spl_list = spl_list[:-(n//2+2)]    

    design_matrix = []
    for i in range(len(spl_list)):
        u = spl_list[i](r)
        design_matrix.append(u)
    design_matrix = np.array(design_matrix).T

    return torch.from_numpy(design_matrix)
    
if __name__ == "__main__":
    ## testing functions bs and pbs
    knots = torch.linspace(start = -math.pi, end = math.pi, steps = 10)
    knots = knots[1:-1]
    boundary_knots = torch.tensor([-math.pi, math.pi])
    
    x = torch.linspace(start = -math.pi, end = math.pi, steps = 200)
    degree = 3

    design_matrix_bs = bs(x, knots, boundary_knots, degree).numpy()
    design_matrix_pbs = pbs(x, knots, boundary_knots, degree).numpy()
    
    fig, axes = plt.subplots()
    for j in range(design_matrix_bs.shape[-1]):
        plt.plot(x, design_matrix_bs[:,j], label = f"{j}", linewidth = 3)
    #plt.legend()
    plt.tight_layout()
    fig.savefig("./output/design_matrix_bs.pdf")
    plt.close()
    
    fig, axes = plt.subplots()
    for j in range(design_matrix_pbs.shape[-1]):
        plt.plot(x, design_matrix_pbs[:,j], label = f"{j}")
    plt.legend()
    plt.tight_layout()
    fig.savefig("./output/design_matrix_pbs.pdf")
    plt.close()
    
    ## test the function bs_lj
    r_min, r_max = 0.3, 2.0
    num_of_basis = 12

    r = torch.linspace(r_min - 0.05, r_max + 1.0, 1000)
    design_matrix, omega = bs_lj(r, r_min, r_max, num_of_basis, True)

    fig, axes = plt.subplots()
    for j in range(design_matrix.shape[-1]):
        plt.plot(r, design_matrix[:,j], label = f"{j}")

    t = np.linspace(r_min, r_max + (r_max - r_min), num_of_basis*2 + 3)
    for i in range(len(t)):
        if t[i] <= r_max:
            plt.axvline(t[i], linestyle = '--')
    plt.legend()
    plt.tight_layout()
    fig.savefig("./output/design_matrix_bs_lj.pdf")
    plt.close()    

    ## test the function bs_rmsd
    r_min, r_max = 0.0, 2.0
    num_of_basis = 12

    r = torch.linspace(0.0, r_max + 1.0, 1000)
    design_matrix = bs_rmsd(r, r_max, num_of_basis)

    fig, axes = plt.subplots()
    for j in range(design_matrix.shape[-1]):
        plt.plot(r, design_matrix[:,j], label = f"{j}")
        
    t = np.linspace(r_min, r_max + (r_max - r_min), num_of_basis*2 + 3)
    for i in range(len(t)):
        if t[i] <= r_max:
            plt.axvline(t[i], linestyle = '--')
    plt.legend()
    plt.tight_layout()
    fig.savefig("./output/design_matrix_bs_rmsd.pdf")
    plt.close()    
    
    
    
