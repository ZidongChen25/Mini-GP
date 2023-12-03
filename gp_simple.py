import numpy as np
import torch
import os
import torch.nn as nn
import unittest
# import kernel as kernel
import kernel_v2 as kernel


class GP_simple(nn.Module):
    def __init__(self, kernel, noise_variance):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = nn.Parameter(torch.tensor([noise_variance]))

    def forward(self, x_train, y_train, x_test):
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train),device=x_train.device)
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)
        
        Kinv_method = 'cholesky1'    # 'direct' or 'cholesky'
        if Kinv_method == 'cholesky1':   # kernel inverse is not stable, use cholesky decomposition instead
            L = torch.cholesky(K)
            L_inv = torch.inverse(L)
            K_inv = L_inv.T @ L_inv
            alpha = K_inv @ y_train
            mu = K_s.T @ alpha
            v = L_inv @ K_s
            cov = K_ss - v.T @ v
        elif Kinv_method == 'cholesky2':
            L = torch.cholesky(K)
            alpha = torch.cholesky_solve(y_train, L)
            mu = K_s.T @ alpha
            v = torch.cholesky_solve(K_s, L)
            cov = K_ss - v.T @ v
        elif Kinv_method == 'direct':
            K_inv = torch.inverse(K)
            mu = K_s.T @ K_inv @ y_train
            cov = K_ss - K_s.T @ K_inv @ K_s
        else:
            raise ValueError('Kinv_method should be either direct or cholesky')
        return mu.squeeze(), cov
            
    def log_likelihood(self, x_train, y_train, use_cholesky=False,Kinv_method = 'torch_distribution_MN1'):
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train),device=x_train.device)
        
        if Kinv_method == 'cholesky1':
            L = torch.cholesky(K)
            L_inv = torch.inverse(L)
            K_inv = L_inv.T @ L_inv
            return -0.5 * (y_train.T @ K_inv @ y_train + torch.logdet(K) + len(x_train) * np.log(2 * np.pi))
        elif Kinv_method == 'cholesky2':
            L = torch.cholesky(K)
            return -0.5 * (y_train.T @ torch.cholesky_solve(y_train, L) + torch.logdet(K) + len(x_train) * np.log(2 * np.pi))
        elif Kinv_method == 'direct':
            K_inv = torch.inverse(K)
            return -0.5 * (y_train.T @ K_inv @ y_train + torch.logdet(K) + len(x_train) * np.log(2 * np.pi))
        elif Kinv_method == 'torch_distribution_MN1':
            L = torch.cholesky(K)
            y_train=y_train.view(y_train.shape[1],y_train.shape[0])
            return torch.distributions.MultivariateNormal(y_train, scale_tril=L).log_prob(y_train)
        elif Kinv_method == 'torch_distribution_MN2':
            y_train=y_train.view(y_train.shape[1],y_train.shape[0])
            return torch.distributions.MultivariateNormal(y_train, K).log_prob(y_train)
        else:
            raise ValueError('Kinv_method should be either direct or cholesky')

class TestGP_simple(unittest.TestCase):
   def test_forward(self):
       # Test forward method
       # ...
       pass

   def test_log_likelihood(self):
       # Test log_likelihood method
       # ...
       pass
       
# downstate here how to use the GP model
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('testing')
    print(torch.__version__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # single output test 1
    torch.manual_seed(1)       #set seed for reproducibility
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(16, 1) * 6
    ytr = torch.sin(xtr) + torch.randn(16, 1) * 0.5 + 10
    
    kernel1 = kernel.ARDKernel(1)
    # kernel1 = kernel.MaternKernel(1)
    # kernel1 = kernel.LinearKernel(1,-1.0,1.)
    
    # kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    
    GPmodel = GP_simple(kernel=kernel1, noise_variance=1.0)
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)

    GPmodel.to(device)
    xtr=xtr.to(device)
    ytr=ytr.to(device)

    for i in range(30):
        optimizer.zero_grad()
        loss = -GPmodel.log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        
    with torch.no_grad():
        xte=xte.to(device)
        ypred, ypred_var = GPmodel.forward(xtr, ytr, xte)
    
    
    plt.errorbar(xte.cpu(), ypred.cpu().reshape(-1).detach(), ypred_var.cpu().diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.plot(xtr.cpu(), ytr.cpu(), 'b+')
    # plt.show()
    output_filename = os.path.join("figures", f"figure1.png")
    plt.savefig(output_filename)
    plt.close()
    
    
    ##test_unit
    # unittest.main()