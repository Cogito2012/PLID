""" Adapted from: https://github.com/nv-research-israel/causal_comp/blob/main/HSIC.py
"""
import math
import torch


def pairwise_distances(x):
    x_distances = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + x_distances + x_distances.t() 

def kernelMatrixGaussian(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    gamma = -1.0 / (sigma ** 2)
    return torch.exp(gamma * pairwise_distances_)

def kernelMatrixLinear(x):
    return torch.matmul(x,x.t())


def _norm(data, dim=-1):
    return data / data.norm(dim=dim, keepdim=True)


def compute_hsic(X, Y, kernelX="Gaussian", kernelY="Gaussian", sigmaX=None, sigmaY=None, norm=True):
    m, D1 = X.shape
    assert(m > 1)
    D2 = Y.shape[-1]
    dtype, device = X.dtype, X.device

    if sigmaX is None or sigmaX is None:
        sigmaX, sigmaY = math.sqrt(D1), math.sqrt(D2)
    
    if norm:
        X, Y = _norm(X), _norm(Y)

    K = kernelMatrixGaussian(X,sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X)
    L = kernelMatrixGaussian(Y,sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y)

    
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.type(dtype).to(device)

    Kc = torch.mm(H,torch.mm(K,H))

    hsic = torch.trace(torch.mm(L,Kc))/((m-1)**2)
    return hsic



if __name__ == '__main__':

    size = (128, 768)
    U = torch.normal(mean=torch.randint(10, 20, size, dtype=torch.float), 
                     std=torch.randint(1, 2, size, dtype=torch.float))
    V1 = torch.normal(mean=torch.randint(-5, 5, size, dtype=torch.float), 
                      std=torch.randint(1, 2, size, dtype=torch.float))
    V2 = 10 * U + 5

    # normalize
    U, V1, V2 = _norm(U), _norm(V1), _norm(V2)

    hsic1 = compute_hsic(U, V1, sigmaX=1, sigmaY=1)
    hsic2 = compute_hsic(U, V2, sigmaX=1, sigmaY=1)
    # hsic1 = compute_hsic(U, V1)
    # hsic2 = compute_hsic(U, V2)
    print(hsic1, hsic2)