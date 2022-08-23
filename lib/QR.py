import numpy as np

def HouseholderQR(A,R,tol = (1e-10)):
  dims = A.Size();
  housedholder_vectors = np.zeros((dims[1]))
  R = A
  for j in range(dims[1]):
  #for (sz j = 0; j < sz(dims[1]); j++) {
    #// compute householder vector
    #housedholder_vectors[j].resize(dims[0] - j);
    housedholder_vectors.append(np.zeros(dims[0] - j))

    #column_mag = R.SubFrobeniusNorm({nm(j), nm(dims[0] - 1)}, {nm(j), nm(j)});
    
    #// serial version
    #/*T
    column_mag=0;
    for i in range(j,dims[0]):
    #for(sz i=j;i<sz(dims[0]);i++)
        column_mag+=R[i,j]*R[i,j];
    
    column_mag=np.sqrt(column_mag)#;*/

    sign = -1 if R(j, j) < T(0) else T(1);
    housedholder_vectors[j][0] = R[j, j] + sign * column_mag;
    D_mag = housedholder_vectors[j,0] * housedholder_vectors[j,0];
    R[j, j] = -sign * column_mag;

    result = T(0);
 		#pragma omp parallel for reduction(+: result)
    #for(nm i=nm(j+1); i<nm(dims[0]);++i){
    for i in range(j+1,dims[0]):
      housedholder_vectors[j,i - j] = R[i, j];
      result += housedholder_vectors[j][i - j] * housedholder_vectors[j][i - j];
      R[i, j] = T(0)

    D_mag += result

    D_mag = np.sqrt(D_mag);
    # serial version
    for i in range(j+1,dims[0]):
    #for(sz i=j+1;i<sz(dims[0]);i++){
        housedholder_vectors[j][i-j]=R[i,j];
        D_mag+=housedholder_vectors[j,i-j]*housedholder_vectors[j,i-j];
        R[i,j]=T(0);
    
    D_mag=np.sqrt(D_mag);
    
    if D_mag > tol:
      #pragma omp parallel for
      for i in range(j,dims[0]):
        housedholder_vectors[j,i - j] /= D_mag
      

      # apply projection to remaining columns of R
      #pragma omp parallel for
      for k in range(j+1,dims[1]):
      #for(nm k=nm(j+1);k<nm(dims[1]);++k){
        dot = (0);
        for i in range(j,dims[0]):
        #for (sz i = j; i < sz(dims[0]); i++) {
          dot += R[i, k] * housedholder_vectors[j,i - j];
        
        for i in range(j,dims[0]):
        #for (sz i = j; i < sz(dims[0]); i++) {
          R[i, k] -= (2) * dot * housedholder_vectors[j,i - j];
        
    return householder_vectors

def ConstructQFromHouseholderVectors(housedholder_vectors, m, l):
  '''
          Input:	m = square matrix size
                          housedholder_vectors: size = n, Ni = housedholder_vectors[i].size() is between 0 and m,
                                          the convention is that the real Householder vector is [0,...,0,housedholder_vectors[i][0],...housedholder_vectors[i][Ni-1]]^T
                          Q = Q0*Q1*...Qn-1
  '''

  n = len(housedholder_vectors);
  #TGSLAssert(n > 0, "TGSL::ALGEBRA::ConstructQFromHouseholderVectors: housedholder_vectors not sized correctly.");

  # set Q to identity initially
  Q = np.zeros(m, l);
  for i in range(l):#(sz i = 0; i < l; i++)
    Q[i,i] = 1.0;
  # multiply Q by the Qk (Householder projections)
  #for (nm k = nm(n - 1); k >= 0; k--) {
  #multiply each Qk times each column (j) of Q
  #TGSLAssert(housedholder_vectors[k].size() <= m, "TGSL::ALGEBRA::ConstructQFromHouseholderVectors: housedholder_vectors not sized correctly.");
  for k in reversed(range(n)):
    i_start = m - housedholder_vectors[k].size();
    for j in range(l):
      dot = T(0)
      for i in range(i_start,m):
        dot += Q[i, j] * housedholder_vectors[k,i - i_start]

      for i in range(i_start,m):
      #for (sz i = i_start; i < m; i++) 
        Q[i, j] -= 2.0 * dot * housedholder_vectors[k,i - i_start]

  return Q
  


def HouseholderQRRec(A, R,tol = 1e-10):
  #std::vector<TV> householder_vectors
  dims = A.Size();
  householder_vectors=HouseholderQR(A,R,tol);
  #if(dims[0]==dims[1]) ConstructQFromHouseholderVectors(householder_vectors,dims[0],Q);
  Q = ConstructQFromHouseholderVectors(householder_vectors,dims[0],dims[1]);

  return Q

def max_non_diag_abs_val(A):
    C = np.abs(A)
    for i in range(len(A)) :
        C[i,i] = 0
    return np.max(C)

def search_max_index(A): 
    C = np.abs(A)
    for i in range(len(A)) :
        C[i,i] = 0

    index=np.argmax(C) 
    index_list=[]

    p = index // len(C)
    q = index % len(C)

    index_list.append(p)
    index_list.append(q)

    return index_list 

def elim_diag(AA,p,q):
    D = np.zeros([len(AA),len(AA)])
    D[:,:] = AA[:,:]


    if AA[p,p]-AA[q,q] ==0 :
        print("hit")
        phi = np.pi/4
    else:
        phi = 0.5*np.arctan(-2*AA[p,q]/(AA[p,p]-AA[q,q]))

    for k in range(len(AA)):
            D[p,k] = AA[p,k]*np.cos(phi) - AA[q,k]*np.sin(phi)
            D[k,p] = D[p,k]    

            D[q,k] = AA[p,k]*np.sin(phi) + AA[q,k]*np.cos(phi)
            D[k,q] = D[q,k]

    D[p,p] = (AA[p,p]+AA[q,q])/2+ ((AA[p,p]-AA[q,q])/2)*np.cos(2*phi)-AA[p,q]*np.sin(2*phi)
    D[q,q] = (AA[p,p]+AA[q,q])/2- ((AA[p,p]-AA[q,q])/2)*np.cos(2*phi)+AA[p,q]*np.sin(2*phi)



    D[p,q] = 0.0
    D[q,p] = 0.0

    return D, phi  

def make_ortho_mat(A,R,p,q,phi): 
    RR = np.identity(len(A))

    RR[p,p] = np.cos(phi)
    RR[p,q] = np.sin(phi)
    RR[q,p] = -np.sin(phi)
    RR[q,q] = np.cos(phi)

    return np.dot(R,RR)
