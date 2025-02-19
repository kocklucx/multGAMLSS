import numpy as np
import pandas as pd
import torch

def etas_to_omegas(etas):
    dim=int(0.5+np.sqrt(2*len(etas[0])+0.25))
    lmb = torch.stack([torch.eye(dim) for _ in range(len(etas))])
    for j in range(len(etas[0])):
        lmb[:,torch.tril_indices(row=dim,col=dim,offset=-1).T[j][0],torch.tril_indices(row=dim,col=dim,offset=-1).T[j][1]]=etas[:,j]
    sigma = torch.inverse(torch.matmul(lmb,torch.transpose(lmb,1,2)))
    normalizer = torch.diag_embed(1/torch.sqrt(torch.diagonal(sigma,offset=0,dim1=1,dim2=2)))
    omegas = torch.matmul(torch.matmul(normalizer,sigma),normalizer)
    return omegas.type(torch.float64)

def scenario_bivariate_normal(n):
    observations = []
    covariables = []
    for _ in range(n):
        x1 = np.random.uniform(1,6,1).item()
        x2 = np.random.uniform(-3,3,1).item()
        x3 = np.random.uniform(-1,1,1).item()
        eta = torch.tensor([np.sin(x1),np.sqrt(x1)*np.cos(x1),np.cos(x2),0.3*x2*np.cos(x2),-np.log(x1)])
        omega = etas_to_omegas(eta[-1:].reshape((1,1)))[0]
        z = np.random.multivariate_normal(mean=np.zeros(2),cov=omega)
        y = np.zeros(2)
        y[0] = torch.distributions.normal.Normal(loc=eta[0],scale=torch.exp(eta[1])).icdf(torch.distributions.normal.Normal(0, 1).cdf(z[0]))
        y[1] = torch.distributions.normal.Normal(loc=eta[2],scale=torch.exp(eta[3])).icdf(torch.distributions.normal.Normal(0, 1).cdf(z[1]))
        observations.append(y)
        covariables.append([x1,x2,x3])
    return np.vstack(observations), np.asarray(covariables)

def scenario_high_dims(n):
    params_marginals = list(np.random.uniform(-1,2,15))
    observations = []
    covariables = []
    for _ in range(n):
        x = np.random.uniform(-0.9,0.9,1).item()
        eta = torch.tensor(params_marginals+[x**2,-x,x**3-x]+list([0 for _ in range(7)]))
        omega = etas_to_omegas(eta[-10:].reshape((1,10)))[0]
        z = np.random.multivariate_normal(mean=np.zeros(5),cov=omega)
        y = np.zeros(5)
        for j in range(5):
            a,b,p = torch.exp(eta[[3*j,3*j+1,3*j+2]])
            y[j]=b*(torch.distributions.normal.Normal(0, 1).cdf(z[j])**(-1/p)-1)**(-1/a)
        observations.append(y)
        covariables.append(x)
    return np.vstack(observations), np.asarray(covariables)

for n in [100,250,500,1000]:
    for j in range(250):
            y,x = scenario_bivariate_normal(n)
            df = pd.DataFrame(y,columns=['y1','y2'])
            df[['x1','x2','x3']]=x
            df.to_csv('data\\bivariate_normal_n'+str(n)+'_'+str(j+1)+'.csv',index=False)

for j in range(250):      
    y,x = scenario_high_dims(500)
    df = pd.DataFrame(y,columns=['y1','y2','y3','y4','y5'])
    df['x']=x
    df.to_csv('data\\high_dims_'+str(j+1)+'.csv',index=False)   

