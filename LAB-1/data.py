import numpy as np

def generate_linear_data(n=100):
    x=np.random.randn(n,2)
    y=(x[:,0]+x[:,1]>0).astype(int)
    return x,y

def generate_nonlinear_data(n=100,noise=0.1):
    radius=np.random.rand(n)
    angle=2*np.pi*np.random.rand(n)

    x1=radius*np.cos(angle)
    x2=radius*np.sin(angle)

    x=np.column_stack((x1,x2))
    y=(radius>0.5).astype(int)

    x+=noise*np.random.randn(n,2)
    return x,y

def generate_xor_data(n=200):
    x=np.random.rand(n,2)
    y=((x[:,0]>0)^(x[:,1]>0)).astype(int)
    return x,y