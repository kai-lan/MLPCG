# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 00:34:19 2021

@author: osman
"""

import numpy as np
import scipy
import scipy.sparse

class pressure_laplacian:
    def __init__(self, n):
        self.n = n
        self.m = (n+1)**2
        self.A = self.pressure_laplacian_matrix()
        self.A_reduced = np.zeros([1,1])
    # creates pressure laplacian for (n+1)-by-(n+1) matrix
    def pressure_laplacian_matrix(self):
        n_ = self.n+1
        n = self.n
        m = self.m
        A = np.zeros([self.m,self.m])
        for i in range(1,self.n):
            A[i][i]=1
            A[i][n_+i]= -1
            A[m-1-i][m-1-i]=1
            A[m-1-i][m-1-i-n_]=-1
            i0 = n_*i
            A[i0][i0]=1
            A[i0][i0+1]=-1
            i1 = n_*i+n
            A[i1][i1]=1
            A[i1][i1-1]=-1
        for i in range(1,n):
            for j in range(1,n):
                k = n_*i+j
                A[k][k]=4
                A[k][k-1]=-1
                A[k][k+1]=-1
                A[k][k+n_]=-1
                A[k][k-n_]=-1
        return A
    
    def create_A_reduced(self):
        reduced_idx = list(range(1,self.n))+list(range(self.n+1,self.n*(self.n+1)))+list(range(self.n*(self.n+1)+1,(self.n+1)**2-1))
        self.A_reduced = self.A[reduced_idx,:]
        self.A_reduced = self.A_reduced[:,reduced_idx]

    def reduced_idx(self):
        return list(range(1,self.n))+list(range(self.n+1,self.n*(self.n+1)))+list(range(self.n*(self.n+1)+1,(self.n+1)**2-1))

    def zero_indexes(self):
        return [0,self.n,self.n*(self.n+1),(self.n+1)**2-1]
        
        
    # here x is np.array with dimension (n+1)^2
    def multiply_pressure_laplacian(self,x):
        return np.matmul(self.A,x)
        
    # here x is np.array with dimension (n+1)^2
    def multiply_pressure_laplacian_fast(self,x):
        y = np.zeros(self.m)
        n = self.n
        n_ = n+1
        for i in range(1,n):
            y[i] = x[i]-x[n_+i]
            y[self.m-1-i] = x[self.m-1-i] - x[self.m-1-i-n]
            i0 = n_*i
            y[i0] = x[i0] - x[i0+1]
            i1 = n_*i+n
            y[i1] = x[i1] - x[i1-1]
            
        for i in range(1,self.n):
            for j in range(1,self.n):
                k = n_*i+j
                y[k] = 4*x[k] - x[k-1] - x[k+1] - x[k+n_] - x[k-n_]
        return y

    
class pressure_laplacian_sparse:
    def __init__(self, n):
        self.n = n
        self.m = (n+1)**2
        self.A_sparse = scipy.sparse.csr_matrix(([1.0], ([1], [1])),[2,2])
        self.create_A_sparse()
        self.A_reduced = np.zeros([1,1])
    # creates pressure laplacian for (n+1)-by-(n+1) matrix

    def create_row_col_and_data(self):
        dim = self.n+1
        data =[]
        rows = []
        cols = []
        for i in range(1,dim-1):
            rows = rows + [i,i]
            cols = cols + [i,i+dim]
            data = data + [1.0,-1.0]

        for j in range(1,dim-1):
            if j%50==0:
                print(j)
            jj = j*dim
            rows = rows + [jj,jj]
            cols = cols + [jj,jj+1]
            data = data + [1.0,-1.0]       
            for i in range(1,dim-1):
                rows = rows + [jj+i,jj+i,jj+i,jj+i,jj+i]
                cols = cols + [jj+i-dim,jj+i-1,jj+i,jj+i+1,jj+i+dim]
                data = data + [-1.0,-1.0,4.0,-1.0,-1.0]
            rows = rows + [jj+dim-1,jj+dim-1]
            cols = cols + [jj+dim-2,jj+dim-1]
            data = data + [-1.0, 1.0]
        for i in range(dim**2-dim+1,dim**2-1):
            rows = rows + [i,i]
            cols = cols + [i-dim,i]
            data = data + [-1.0,1.0]
        return rows, cols, data

    def zero_indexes(self):
        return [0,self.n,self.n*(self.n+1),(self.n+1)**2-1]

    def create_A_sparse(self):
        rows, cols, data = self.create_row_col_and_data()    
        self.A_sparse = scipy.sparse.csr_matrix((data, (rows, cols)),[self.m,self.m])
         
    def create_indices_and_values(self):
        dim = int(np.sqrt(self.n))
        indices = []
        values = []
        for i in range(1,dim-1):
            indices = indices + [[i,i]]
            values = values + [1.0]
            indices = indices + [[i,i+dim]]
            values = values + [-1.0]
        for j in range(1,dim-1):
            jj = j*dim
            indices = indices + [[jj,jj]]    
            values = values + [1.0]
            indices = indices + [[jj,jj+1]]
            values = values + [-1.0]
            for i in range(1,dim-1):
                indices = indices + [[jj+i,jj+i-dim]]
                values = values + [-1.0]
                indices = indices + [[jj+i,jj+i-1]]
                values = values + [-1.0] 
                indices = indices + [[jj+i,jj+i]]
                values = values + [4.0]
                indices = indices + [[jj+i,jj+i+1]]
                values = values + [-1.0]
                indices = indices + [[jj+i,jj+i+dim]]
                values = values + [-1.0] 
            indices = indices + [[jj+dim-1,jj+dim-2]]
            values = values + [-1.0]
            indices = indices + [[jj+dim-1,jj+dim-1]]
            values = values + [1.0]
        for i in range(dim**2-dim+1,dim**2-1):
            indices = indices + [[i,i-dim]]
            values = values + [-1.0]
            indices = indices + [[i,i]]
            values = values + [1.0]
        return indices, values
       
        
    # here x is np.array with dimension (n+1)^2
    def multiply_pressure_laplacian(self,x):
        return np.matmul(self.A,x)
#from numba import njit, prange

class pressure_laplacian_3D_sparse:
    def __init__(self, n):
        self.n = n
        self.m = (n+1)**3
        self.A_sparse = scipy.sparse.csr_matrix(([1.0], ([1], [1])),[2,2])
        self.create_A_sparse()
    # creates pressure laplacian for (n+1)-by-(n+1) matrix




    def create_row_col_and_data(self):    
        
        d = self.n+1
        d2=d**2
        data =[]
        rows = []
        cols = []
        for i in range(1,d-1):
            for j in range(1,d-1):
                s = d*i+j
                rows = rows + [s, s]
                cols = cols + [s, s + d2]
                data = data + [1.0, -1.0]

        for k in range(1,d-1):
            if k%10==0:
                print(k)
            
            kk = k*d2
            for j in range(1,d-1):
                s = kk+j
                rows = rows + [s, s]
                cols = cols + [s, s + d]
                data = data + [1.0, -1.0]
            
            for i in range(1,d-1):
                ii = d*i
                s = kk+ii
                rows = rows + [s, s]
                cols = cols + [s, s + 1]
                data = data + [1.0, -1.0]
                for j in range(1,d-1):
                    s = kk+ii+j
                    rows = rows + [s, s, s, s, s, s, s]
                    cols = cols + [s-d2, s-d, s-1,s,s+1,s+d,s+d2]
                    data = data + [-1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0]
                
                s = kk+ii+d-1
                rows = rows + [s, s]
                cols = cols + [s-1, s]
                data = data + [-1.0, 1.0]
            
            for j in range(1,d-1):
                s = kk+d2-d+j
                rows = rows + [s, s]
                cols = cols + [s-d, s]
                data = data + [-1.0, 1.0]
                        
        for i in range(1,d-1):
            for j in range(1,d-1):
                s = d**3 - d2 + d*i + j
                rows = rows + [s, s]
                cols = cols + [s-d2, s]
                data = data + [-1.0, 1.0]          
            
        return rows, cols, data

    #@njit(parallel=True)
    def create_row_col_and_data_parallel(self):
        d = self.n+1
        d2=d**2
        u = d-2
        num_rows = 4*u**2 + u*(4*u + u*(4+ 7*u))
        data =[0]*num_rows
        rows = [0]*num_rows
        cols = [0]*num_rows
        nn = 0
        for i in range(1,d-1):
            for j in range(1,d-1):
                s = d*i+j
                rows[nn:nn+2] = [s, s]
                cols[nn:nn+2] = [s, s + d2]
                data[nn:nn+2] = [1.0, -1.0]
                nn = nn+2

        for k in range(1,d-1):
            if k%10==0:
                print(k)
            
            kk = k*d2
            for j in range(1,d-1):
                s = kk+j
                rows[nn:nn+2] = [s, s]
                cols[nn:nn+2] = [s, s + d]
                data[nn:nn+2] = [1.0, -1.0]
                nn = nn+2
            
            for i in range(1,d-1):
                ii = d*i
                s = kk+ii
                rows[nn:nn+2] = [s, s]
                cols[nn:nn+2] = [s, s + 1]
                data[nn:nn+2] = [1.0, -1.0]
                nn = nn+2
                for j in range(1,d-1):
                    s = kk+ii+j
                    rows[nn:nn+7] = [s, s, s, s, s, s, s]
                    cols[nn:nn+7] = [s-d2, s-d, s-1,s,s+1,s+d,s+d2]
                    data[nn:nn+7] = [-1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0]
                    nn = nn+7
                
                s = kk+ii+d-1
                rows[nn:nn+2] = [s, s]
                cols[nn:nn+2] = [s-1, s]
                data[nn:nn+2] = [-1.0, 1.0]
                nn = nn+2
            
            for j in range(1,d-1):
                s = kk+d2-d+j
                rows[nn:nn+2] = [s, s]
                cols[nn:nn+2] = [s-d, s]
                data[nn:nn+2] = [-1.0, 1.0]
                nn = nn+2 
        for i in range(1,d-1):
            for j in range(1,d-1):
                s = d**3 - d2 + d*i + j
                rows[nn:nn+2] = [s, s]
                cols[nn:nn+2] = [s-d2, s]
                data[nn:nn+2] = [-1.0, 1.0]          
                nn = nn+2 
        return rows, cols, data    


    def zero_indexes(self):
        d = self.n+1
        d2=d**2
        d3 = d**3
        z = list(range(0,d))
        
        for i in range(1,d-1):
            z = z+[d*i,d*(i+1)-1] 
        
        z = z + list(range(d2-d,d2))

        for k in range(1,d-1):            
            kk = k*d2
            z = z+[kk,kk+d-1,kk+d2-d, kk+d2-1]
                        
        z = z+list(range(d2*(d-1),d2*(d-1)+d))
        
        for i in range(1,d-1):
            z = z+[d2*(d-1) + d*i, d2*(d-1) + d*(i+1)-1] 
        
        z = z + list(range(d3-d,d3))
        return z

    
    def create_A_sparse(self):
        rows, cols, data = self.create_row_col_and_data()    
        #rows, cols, data = self.create_row_col_and_data_parallel()  
        self.A_sparse = scipy.sparse.csr_matrix((data, (rows, cols)),[self.m,self.m])
         
    def create_indices_and_values(self):
        dim = int(np.sqrt(self.n))
        indices = []
        values = []
        for i in range(1,dim-1):
            indices = indices + [[i,i]]
            values = values + [1.0]
            indices = indices + [[i,i+dim]]
            values = values + [-1.0]
        for j in range(1,dim-1):
            jj = j*dim
            indices = indices + [[jj,jj]]    
            values = values + [1.0]
            indices = indices + [[jj,jj+1]]
            values = values + [-1.0]
            for i in range(1,dim-1):
                indices = indices + [[jj+i,jj+i-dim]]
                values = values + [-1.0]
                indices = indices + [[jj+i,jj+i-1]]
                values = values + [-1.0] 
                indices = indices + [[jj+i,jj+i]]
                values = values + [4.0]
                indices = indices + [[jj+i,jj+i+1]]
                values = values + [-1.0]
                indices = indices + [[jj+i,jj+i+dim]]
                values = values + [-1.0] 
            indices = indices + [[jj+dim-1,jj+dim-2]]
            values = values + [-1.0]
            indices = indices + [[jj+dim-1,jj+dim-1]]
            values = values + [1.0]
        for i in range(dim**2-dim+1,dim**2-1):
            indices = indices + [[i,i-dim]]
            values = values + [-1.0]
            indices = indices + [[i,i]]
            values = values + [1.0]
        return indices, values
        
    

        
        
    # here x is np.array with dimension (n+1)^2
    def multiply_pressure_laplacian(self,x):
        return np.matmul(self.A,x)
    
    
