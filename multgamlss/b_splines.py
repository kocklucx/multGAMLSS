import torch
import numpy as np

def set_knots(anz_splines,x):
    dist = (max(x)-min(x))/(anz_splines-3)
    knots = np.linspace(min(x)-3*dist,max(x)+3*dist,anz_splines+4)
    return knots

def design_matrix(knots,x,deg=3):
    design_matrix=torch.zeros((len(x),len(knots)-deg-1))
    for i in range(len(x)):
        for j in range(len(knots)-deg-1):
            design_matrix[i][j]=spline(t=j,deg=deg,knots=knots,x=x[i])
    return design_matrix.type(torch.float64)

def pen_matrix_first_order(dim):
    K=np.zeros((dim,dim))
    K[np.arange(dim-1),np.arange(dim-1)+1]=-np.ones(dim-1)
    K[np.arange(dim-1)+1,np.arange(dim-1)]=-np.ones(dim-1)
    K[np.arange(dim),np.arange(dim)]=2*np.ones(dim)
    K[0,0]=1
    K[dim-1,dim-1]=1
    return torch.tensor(K) 

def pen_matrix(dim): #second order
    K=np.diag(6*np.ones(dim),0)+np.diag(-4*np.ones(dim-1),-1)+np.diag(np.ones(dim-2),-2)+np.diag(-4*np.ones(dim-1),1)+np.diag(np.ones(dim-2),2)
    K[0,[0,1]]=[1,-2]
    K[1,[0,1,2]]=[-2,5,-4]
    K[-1,[-2,-1]]=[-2,1]
    K[-2,[-2,-1]]=[5,-2]
    threshold = 1e-5
    eps = 1/10
    eigvalues, eigvectors = np.linalg.eigh(K)
    eigvalues[eigvalues<threshold]=eps*np.min(eigvalues[eigvalues>=threshold])
    K=eigvectors@np.diag(eigvalues)@np.linalg.inv(eigvectors)
    return torch.tensor(K) 

def spline(t,deg,knots,x):
    # t index of spline
    # deg is degree
    # knots ordered list of knots
    # x: point of evaluation
    if deg==0:
        if x>=knots[t] and x<knots[t+1]:
            return 1
        else:
            return 0
    else:
        return (x-knots[t])/(knots[t+deg]-knots[t])*spline(t,deg-1,knots,x)+(knots[t+deg+1]-x)/(knots[t+deg+1]-knots[t+1])*spline(t+1,deg-1,knots,x)

def cyclic_design_matrix(knots,x,deg=3):
    # generates a cyclic spline basis for values between knots[0] and knots[-1]. By default knots[0]=knots[-1] is where the cyclce happens
    length = knots[-1]-knots[0]
    pseudo_knots=np.asarray([knots[-3]-length,knots[-2]-length]+list(knots)+[knots[1]+length])
    design_matrix=torch.zeros((len(x),len(pseudo_knots)-deg-1))
    for i in range(len(x)):
        for j in range(len(pseudo_knots)-deg-1):
            design_matrix[i][j]=spline(t=j,deg=deg,knots=pseudo_knots,x=x[i])+spline(t=j,deg=deg,knots=pseudo_knots,x=x[i]-length)+spline(t=j,deg=deg,knots=pseudo_knots,x=x[i]+length)
    return design_matrix.type(torch.float64)

def cyclic_pen_matrix(dim):
    K=np.diag(6*np.ones(dim),0)+np.diag(-4*np.ones(dim-1),-1)+np.diag(np.ones(dim-2),-2)+np.diag(-4*np.ones(dim-1),1)+np.diag(np.ones(dim-2),2)
    K[0,dim-1]=-4
    K[dim-1,0]=-4
    K[dim-2,0]=1
    K[dim-1,1]=1
    K[0,dim-2]=1
    K[1,dim-1]=1
    threshold = 1e-5
    eps = 1/10
    eigvalues, eigvectors = np.linalg.eigh(K)
    eigvalues[eigvalues<threshold]=eps*np.min(eigvalues[eigvalues>=threshold])
    K=eigvectors@np.diag(eigvalues)@np.linalg.inv(eigvectors)
    return torch.tensor(K) 