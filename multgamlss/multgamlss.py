# multgamlss code to reanalyze the two simulated data sets considered in Kock and Klein (2023). 
# The main function sampler(design_matrices,penalty_matrices,y,chain_length) generates chain_length samples from the posterior
# distribution. The inputs are:
#       design_matrices: A list of k lists, where the k-th list contains the design matrices for the k-th
#            predictor. You may need to preprocess your data and create the appropriate design matrices based 
#            on the structure of your predictors.
#       penalty_matrices: A list with the same structure as design_matrices containing the corresponding
#            penalty matrices
#       y: This variable represents the response variables in your regression model. It should be a tensor where each row 
#            corresponds to an observation and each column corresponds to a response variable.
# Note that the distributional assumptions on the marginal distributions might need to be addappted manually by changing
# the functions log_pdf_marginal(etas,y,index) and cdf_marginal(etas,y,index) calculating the log probability density and 
# cumulative distribution function for the marginal distributions of the response variables. The inputs are 
#       etas: A k times n dimensional tensor, where entry kn corresponds to the k-th linear predictor for observation n
#       index: The index j=1,...,d for which the log-pdf / cdf should be calculated
#       y: The observation matrix.

import torch
import numpy as np
import copy
from b_splines import set_knots, design_matrix, pen_matrix
import pandas as pd
import pickle

def log_target(betas_effects,betas_intercepts,tau2s,design_matrices,y,penalty_matrices):
    #Calculates the log of the joint target density, which includes the log likelihood and the log prior.
    etas = beta_to_eta(y,design_matrices,betas_effects,betas_intercepts)
    targ = log_likelihood(y,etas).sum() + log_prior(tau2s,betas_effects,betas_intercepts,penalty_matrices)
    return targ

def log_pdf_marginal(etas,y,index): 
    ##Computes the log probability density for the 
    ##marginal distributions of the response variables. Note that this has to be adapted manually 
    ## Five dimensional example, dagum marginals
    #a = torch.exp(etas[:,3*index])
    #b = torch.exp(etas[:,3*index+1])
    #p = torch.exp(etas[:,3*index+2])
    #return torch.log(a)+torch.log(p)-torch.log(y[:,index])+a*p*torch.log(y[:,index]/b)-(p+1)*torch.log((y[:,index]/b).pow(a)+1)
    # bivariate normal example, normal marginals
    mu = etas[:,2*index]
    sigma = torch.exp(etas[:,2*index+1])
    return torch.distributions.normal.Normal(mu, sigma).log_prob(y[:,index])

    
def cdf_marginal(etas,y,index): 
    ##Computes cumulative distribution function for the 
    ##marginal distributions of the response variables. Note that this has to be adapted manually 
    eps=1e-5
    ## Five dimensional example, dagum marginals
    #a = torch.exp(etas[:,3*index])
    #b = torch.exp(etas[:,3*index+1])
    #p = torch.exp(etas[:,3*index+2])
    #return (1+(y[:,index]/b).pow(-a)).pow(-p).clamp(eps,1-eps)
    # bivariate normal example, normal marginals
    mu = etas[:,2*index]
    sigma = torch.exp(etas[:,2*index+1])
    return torch.distributions.normal.Normal(mu, sigma).cdf(y[:,index]).clamp(eps,1-eps)

def log_likelihood(y,etas):
    #Computes the log likelihood of the observations given the predictors eta.
    dim = len(y[0])
    num_obs = len(y)
    ll=0
    for j in range(dim):
        ll += log_pdf_marginal(etas,y,j) 
    u = torch.distributions.normal.Normal(0, 1).icdf(torch.stack([cdf_marginal(etas,y,j) for j in range(dim)],axis=1)) 
    omegas = etas_to_omegas(etas[:,-int((dim*(dim-1))/2):])
    ll+= -0.5*torch.matmul(torch.matmul(u.view(num_obs,1,dim),torch.inverse(omegas)-torch.eye(dim)).view(num_obs,1,dim), u.view(num_obs,dim, 1)).reshape(-1)-0.5*torch.log(torch.det(omegas))
    return ll

def mean_var_proposal_constant(betas_effects,betas_intercepts,tau2s,y,design_matrices,index_eta):
    # Calculates the mean and variance of the proposal distribution for updating the intercept term
    etas = beta_to_eta(y,design_matrices,betas_effects,betas_intercepts)
    eta = copy.deepcopy(etas[:,index_eta])
    eta_ohne = eta - betas_intercepts[index_eta]
    eta.requires_grad=True
    etas[:,index_eta]=eta
    l =  log_likelihood(y,etas).sum()
    v = torch.autograd.grad(l, eta, create_graph=True)[0]
    w= -torch.autograd.grad(v.sum(), eta)[0]
    with torch.no_grad():
        w[w<min(w[w>0])]=0.01*min(w[w>0])
        z = eta + v/w
        var = 1/w.sum()
        mean = (var*w*(z-eta_ohne)).sum()
    return mean, var

def proposal_mean_cov(betas_effects,betas_intercepts,tau2s,y,design_matrices,penalty_matrices,index_eta,index_beta):
    # Calculates the mean and covariance of the proposal distribution for updating the spline coefficients.
    etas = beta_to_eta(y,design_matrices,betas_effects,betas_intercepts)
    eta = copy.deepcopy(etas[:,index_eta])
    eta_ohne = eta - torch.matmul(design_matrices[index_eta][index_beta],betas_effects[index_eta][index_beta])
    eta.requires_grad=True
    etas[:,index_eta]=eta
    l =  log_likelihood(y,etas).sum()
    v = torch.autograd.grad(l, eta, create_graph=True)[0]
    w= -torch.autograd.grad(v.sum(), eta)[0]
    with torch.no_grad():
        w[w<min(w[w>0])]=0.01*min(w[w>0]) #avoid cholesky error when sampling
        z = eta + v/w
        P_inv = torch.inverse(torch.matmul(torch.matmul(design_matrices[index_eta][index_beta].T,torch.diag(w).type(torch.float64)),design_matrices[index_eta][index_beta])+1/tau2s[index_eta][index_beta]*penalty_matrices[index_eta][index_beta])
        mu = torch.matmul(torch.matmul(torch.matmul(P_inv,design_matrices[index_eta][index_beta].T),torch.diag(w).type(torch.float64)),z-eta_ohne)
    return mu, P_inv

def log_prior(tau2s,betas_effects,betas_intercepts,penalty_matrices,a_tau2=0.001,b_tau2=0.001):
    #Computes the log prior probability of the model's parameters.
    prior = 0
    for p in range(len(betas_effects)):
        for j in range(len(betas_effects[p])):
            prior+=-0.5/tau2s[p][j]*torch.matmul(betas_effects[p][j].T,torch.matmul(penalty_matrices[p][j],betas_effects[p][j]))
            prior += ((-a_tau2-1)*torch.log(tau2s[p][j])-b_tau2/tau2s[p][j]).sum() # prior for tau2
    return prior[0]

def etas_to_omegas(etas):
    #Converts the spline coefficients to a precision matrix
    dim=int(0.5+np.sqrt(2*len(etas[0])+0.25))
    lmb = torch.stack([torch.eye(dim) for _ in range(len(etas))])
    for j in range(len(etas[0])):
        lmb[:,torch.tril_indices(row=dim,col=dim,offset=-1).T[j][0],torch.tril_indices(row=dim,col=dim,offset=-1).T[j][1]]=etas[:,j]
    sigma = torch.inverse(torch.matmul(lmb,torch.transpose(lmb,1,2)))
    normalizer = torch.diag_embed(1/torch.sqrt(torch.diagonal(sigma,offset=0,dim1=1,dim2=2)))
    omegas = torch.matmul(torch.matmul(normalizer,sigma),normalizer)
    return omegas.type(torch.float64)

def beta_to_eta(y,design_matrices,betas_effects,betas_intercepts): 
    #Converts the spline coefficients to the linear predictor.
    etas=betas_intercepts*torch.ones(len(y),len(design_matrices))
    for j in range(len(design_matrices)):
        for i in range(len(design_matrices[j])):
            etas[:,j]+=torch.matmul(design_matrices[j][i],betas_effects[j][i])
    return etas
 
def update_constant(betas_effects,betas_intercepts,tau2s,y,design_matrices,penalty_matrices,index_eta):
    #Updates the intercept term using a Metropolis-Hastings step
    m_old, var_old = mean_var_proposal_constant(betas_effects,betas_intercepts,tau2s,y,design_matrices,index_eta)
    betas_intercepts_star = copy.deepcopy(betas_intercepts)
    betas_intercepts_star[index_eta] = torch.normal(m_old,np.sqrt(var_old)).item()
    m_star, var_star = mean_var_proposal_constant(betas_effects,betas_intercepts_star,tau2s,y,design_matrices,index_eta)
    if np.log(1-np.random.random()) <= log_target(betas_effects,betas_intercepts_star,tau2s,design_matrices,y,penalty_matrices)-log_target(betas_effects,betas_intercepts,tau2s,design_matrices,y,penalty_matrices)+torch.distributions.normal.Normal(m_star,torch.sqrt(var_star)).log_prob(betas_intercepts[index_eta])-torch.distributions.normal.Normal(m_old,torch.sqrt(var_old)).log_prob(betas_intercepts_star[index_eta]):
        return betas_intercepts_star[index_eta]
    else:
        return betas_intercepts[index_eta]

def proposal_sample(mean,cov):
    #samples from a normal distribution with mean mean and covariance matrix cov under 1\top b=0
    a = torch.ones(len(mean)).type(torch.float64)
    sample = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov).sample()
    return sample-1/torch.matmul(a,torch.matmul(cov,a))*torch.matmul(a,cov.T)*torch.matmul(a,sample)

def proposal_log_density(x,mean,cov):
    #returns the log density of a normal distribution with mean mean and covariance matrix cov under restirction 1\top b=0
    # works only if 1\top x = 0, otherwise it should return 0
    return torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov).log_prob(x)-0.5-torch.distributions.normal.Normal(loc=mean.sum(), scale=torch.sqrt(cov.sum())).log_prob(0)

def find_map(design_matrices,penalty_matrices,y,anz_steps=1000):
    #Finds the maximum a posteriori (MAP) estimates of the model's parameters using
    # gradient descent optimization
    dim_eta = len(design_matrices)
    betas_effects = [[torch.zeros(len(X[0]),dtype=torch.float64) for X in design_matrices[j]] for j in range(len(design_matrices))]
    betas_intercepts = torch.zeros(dim_eta)
    tau2s =  [[torch.ones(1) for X in design_matrices[j]] for j in range(len(design_matrices))]
    params_to_optimize = []
    for s in tau2s:
        for t in s:
            t.requires_grad=True
            params_to_optimize.append(t)
    for s in betas_effects:
        for t in s:
            t.requires_grad=True
            params_to_optimize.append(t)
    betas_intercepts.requires_grad = True
    params_to_optimize.append(betas_intercepts)
    optimizer = torch.optim.Adam(params_to_optimize,lr=0.05)
    for _ in range(anz_steps):
        optimizer.zero_grad()
        ln_target = -log_target(betas_effects,betas_intercepts,tau2s,design_matrices,y,penalty_matrices)
        ln_target.backward()
        optimizer.step()
        with torch.no_grad():
            for p in range(dim_eta):
                for j in range(len(design_matrices[p])):
                    tau2s[p][j]=tau2s[p][j].clamp(1e-5)
    for par in params_to_optimize:
        par.requires_grad = False
    for p in range(dim_eta):
        for j in range(len(design_matrices[p])):
            betas_intercepts[p] += betas_effects[p][j].sum().item()/len(betas_effects[p][j])
            betas_effects[p][j] -= betas_effects[p][j].sum()/len(betas_effects[p][j])
            a_tau2=0.001
            b_tau2=0.001
            tau2s[p][j] = torch.tensor([1/np.random.gamma(np.linalg.matrix_rank(penalty_matrices[p][j])/2+a_tau2,1/(0.5*torch.matmul(torch.matmul(betas_effects[p][j].T,penalty_matrices[p][j]),betas_effects[p][j])+b_tau2))])
    print('Found MAP',-ln_target.item())
    return betas_effects, betas_intercepts, tau2s

def sampler(design_matrices,penalty_matrices,y,chain_length):
    #Performs MCMC sampling to obtain posterior samples
    dim_eta = len(design_matrices)
    a_tau2=0.001
    b_tau2=0.001
    betas_effects, betas_intercepts, tau2s = find_map(design_matrices,penalty_matrices,y) 
    betas_effects_draws = [betas_effects]
    betas_intercepts_draws = [betas_intercepts]
    tau2_draws = [tau2s]    
    for step in range(chain_length):
        for p in range(dim_eta):
            betas_intercepts[p]=update_constant(betas_effects,betas_intercepts,tau2s,y,design_matrices,penalty_matrices,p)
            for j in range(len(design_matrices[p])):
                m_old, s_old = proposal_mean_cov(betas_effects,betas_intercepts,tau2s,y,design_matrices,penalty_matrices,p,j)
                betas_effects_star = copy.deepcopy(betas_effects)
                betas_effects_star[p][j] = proposal_sample(m_old,s_old)
                m_star, s_star = proposal_mean_cov(betas_effects_star,betas_intercepts,tau2s,y,design_matrices,penalty_matrices,p,j)
                if np.log(1-np.random.random()) <= log_target(betas_effects_star,betas_intercepts,tau2s,design_matrices,y,penalty_matrices)-log_target(betas_effects,betas_intercepts,tau2s,design_matrices,y,penalty_matrices)+proposal_log_density(betas_effects[p][j],m_star,s_star)-proposal_log_density(betas_effects_star[p][j],m_old,s_old):
                    betas_effects = betas_effects_star
                tau2s[p][j] = torch.tensor([1/np.random.gamma(np.linalg.matrix_rank(penalty_matrices[p][j])/2+a_tau2,1/(0.5*torch.matmul(torch.matmul(betas_effects[p][j].T,penalty_matrices[p][j]),betas_effects[p][j])+b_tau2))])
        betas_effects_draws.append(copy.deepcopy(betas_effects))
        betas_intercepts_draws.append(copy.deepcopy(betas_intercepts))
        tau2_draws.append(copy.deepcopy(tau2s))
    return betas_effects_draws, betas_intercepts_draws, tau2_draws

# bivariate normal example
anz_splines = 22
knots1 = set_knots(anz_splines,[1,6])
knots2 = set_knots(anz_splines,[-3,3])
knots3 = set_knots(anz_splines,[-1,1])
pen = pen_matrix(anz_splines)
for j in range(250):
    for n in [100,250,500,1000]:
        df = pd.read_csv('data\\bivariate_normal_n'+str(n)+'_'+str(j+1)+'.csv')
        x = torch.tensor(df[['x1','x2','x3']].values)
        y = torch.tensor(df[['y1','y2']].values)
        design_x1 = design_matrix(knots1,x[:,0],deg=3)
        design_x2 = design_matrix(knots2,x[:,1],deg=3)
        design_x3 = design_matrix(knots3,x[:,2],deg=3)
        design_matrices = [[design_x1, design_x2, design_x3]  for _ in range(5)]
        penalty_matrices = [[pen, pen, pen] for _ in range(5)]
        betas_effects_draws, betas_intercepts_draws, tau2_draws = sampler(design_matrices,penalty_matrices,y,chain_length=5555)
        results = {'betas_effects_draws': betas_effects_draws,'betas_intercepts_draws': betas_intercepts_draws,'tau2_draws': tau2_draws}
        pickle.dump(results, open(r'multgamlss\multgamlss_bivariate_normal_n'+str(n)+'_'+str(j+1)+'.p', "wb" ))

# five dimensional non-Gaussian example
#anz_splines = 22
#knots = set_knots(anz_splines,[-0.9,0.9])
#pen = pen_matrix(anz_splines)
#for j in range(250):
#    df = pd.read_csv('data\\high_dims_'+str(j+1)+'.csv')
#    x = torch.tensor(df['x'].values)
#    y = torch.tensor(df[['y1','y2','y3','y4','y5']].values)
#    design = design_matrix(knots,x,deg=3)
#    design_matrices = [[] for _ in range(15)]+[[design]  for _ in range(10)]
#    penalty_matrices = [[] for _ in range(15)]+[[pen]  for _ in range(10)]
#    betas_effects_draws, betas_intercepts_draws, tau2_draws = sampler(design_matrices,penalty_matrices,y,chain_length=5555)
#    results = {'betas_effects_draws': betas_effects_draws,'betas_intercepts_draws': betas_intercepts_draws,'tau2_draws': tau2_draws}
#    pickle.dump(results, open(r'multgamlss\multgamlss_high_dims_'+str(j+1)+'.p', "wb" ))