import torch
import numpy as np
from scipy import optimize

def contrastive_learning(log_q_noise, log_q_data,
                         basis_noise, basis_data,
                         options = {'disp': True,
                                    'gtol': 1e-5}
):

    """
    Contrastive learning coefficients

    Parameters
    ----------
    log_q_noise: 1-dimensional tensor
        the logrithm of probability density for noise data under the noise distribution
    log_q_data: 1-dimensional tensor
        the logrithm of probability density for target data under the noise distribution
    basis_noise: 2-dimensional tensor
        the design matrix contraining basis values of noise data for compute the logrithm of probablity density for the target distribution
    basis_data: 2-dimensional tensor
        the design matrix contraining basis values of target data for compute the logrithm of probablity density for the target distribution

    Returns
    -------
    alpha: 
    
    """
    
    assert(basis_noise.shape[-1] == basis_data.shape[-1])
    assert(len(log_q_noise) == basis_noise.shape[0])
    assert(len(log_q_data) == basis_data.shape[0])

    basis_size = basis_noise.shape[-1]
    alphas = torch.zeros(basis_size, dtype=torch.float64)
    F = torch.zeros(1, dtype=torch.float64)

    x_init = np.concatenate([alphas.data.numpy(), F])
    
    def compute_loss_and_grad(x):
        alphas = torch.tensor(x[0:basis_size], requires_grad = True)
        F = torch.tensor(x[-1], requires_grad = True)

        u_data = torch.matmul(basis_data, alphas)
        u_noise = torch.matmul(basis_noise, alphas)

        num_samples_p = basis_data.shape[0]
        num_samples_q = basis_noise.shape[0]

        nu = F.new_tensor([num_samples_q / num_samples_p])
        
        log_p_data = - (u_data - F) - torch.log(nu)
        log_p_noise = - (u_noise - F) - torch.log(nu)

        log_q = torch.cat([log_q_noise, log_q_data])
        log_p = torch.cat([log_p_noise, log_p_data])

        logit = log_p - log_q
        target = torch.cat([torch.zeros_like(log_q_noise),
                            torch.ones_like(log_q_data)])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, target)               
        loss.backward()
        
        grad = torch.cat([alphas.grad, F.grad[None]]).numpy()

        return loss.item(), grad

    loss, grad = compute_loss_and_grad(x_init)
    
    
    results = optimize.minimize(compute_loss_and_grad,
                                x_init,
                                jac=True,
                                method='L-BFGS-B',
                                options = options)
    x = results['x']
    
    # x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
    #                                  x_init,
    #                                  iprint = 1,
    #                                  pgtol = 1e-6,
    #                                  factr = 100)

    alphas = x[0:basis_size]
    F = x[-1]

    return torch.from_numpy(alphas), F

def contrastive_learning_numpy(log_q_noise, log_q_data,
                               basis_noise, basis_data):

    """
    Contrastive learning coefficients

    Parameters
    ----------
    log_q_noise: 1-dimensional array
        the logrithm of probability density for noise data under the noise distribution
    log_q_data: 1-dimensional array
        the logrithm of probability density for target data under the noise distribution
    basis_noise: 2-dimensional array
        the design matrix contraining basis values of noise data for compute the logrithm of probablity density for the target distribution
    basis_data: 2-dimensional array
        the design matrix contraining basis values of target data for compute the logrithm of probablity density for the target distribution

    Returns
    -------
    alpha: 
    
    """
    
    assert(basis_noise.shape[-1] == basis_data.shape[-1])
    assert(len(log_q_noise) == basis_noise.shape[0])
    assert(len(log_q_data) == basis_data.shape[0])

    basis_size = basis_noise.shape[-1]
    alphas = np.zeros(basis_size)
    F = np.zeros(1)
    
    x_init = np.concatenate([alphas, F])

    log_q = np.concatenate([log_q_noise, log_q_data])
    y = np.concatenate([np.zeros_like(log_q_noise),
                        np.ones_like(log_q_data)])

    basis = np.concatenate([basis_noise, basis_data])

    num_samples_p = basis_data.shape[0]
    num_samples_q = basis_noise.shape[0]

    log_nu = np.log(num_samples_q / float(num_samples_p))
    
    def compute_loss_and_grad(x):
        alphas = x[0:basis_size]
        F = x[-1]

        ## compute loss = -(y*h - np.log(1 + np.exp(h)))
        h = -(np.matmul(basis, alphas) - F) - log_q - log_nu
        loss = -(y*h - (np.maximum(h, 0) + np.log(1 + np.exp(-np.abs(h)))))
        loss = np.mean(loss, 0)

        ## compute gradients
        p = 1. / (1 + np.exp(-np.abs(h)))
        p[h < 0] = 1 - p[h < 0]

        grad_alphas = np.matmul(basis.T, y - p) / y.shape[0]
        grad_F = -np.mean(y - p, keepdims = True)
        
        grad = np.concatenate([grad_alphas, grad_F])

        return loss, grad

    loss, grad = compute_loss_and_grad(x_init)
    x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                     x_init,
                                     iprint = 1)

    alphas = x[0:basis_size]
    F = x[-1]

    return alphas, F
