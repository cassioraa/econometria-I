import numpy as np 

from numpy.linalg import inv as inv
from numpy.linalg import det as det

#------------------------------
# state space representation
#------------------------------

def state_space(x):

    ''' 
        y(t) = Λ s(t) + u(t),     u(t) ~ N(0, R)
        s(t) = Φ s(t-1) + η(t),   η(t) ~ N(0, Q)
    y(t): (nvar x 1)
    s(t): (nstate x 1)
    Λ: (nvar x nstate)
    Φ: (nstate x nstate)
    R: (nvar x nvar)
    Q: (nstate x nstate) 
    '''


    Λ = np.array([[1,0]])
    
    Φ = np.array([[1, 1],
                  [0, 1]])
    
    R = np.array([[x[0]**2]])
    
    Q = np.array([[0, 0],
                  [0, x[1]**2]])

    return (Λ, Φ, R, Q)


#------------------
# kalman filter
#------------------

def kalman(Λ, Φ, R, Q, y):
    global st
    '''
        y(t) = Λ s(t) + u(t),     u(t) ~ N(0, R)
        s(t) = Φ s(t-1) + η(t),   η(t) ~ N(0, Q)
    
    y(t): (nvar x 1)
    s(t): (nstate x 1)
    Λ: (nvar x nstate)
    Φ: (nstate x nstate)
    R: (nvar x nvar)
    Q: (nstate x nstate) 
    '''

    # get dimensions
    nvar, nobs = np.shape(y) 
    nstates = len(Q)
    
    # arrays to store the results
    lnl = np.zeros((nobs,))
    s10 = np.zeros((nstates, nobs))
    p10 = np.zeros((nstates, nstates, nobs))
    s11 = np.zeros((nstates, nobs))
    p11 = np.zeros((nstates, nstates, nobs))
    ss  = np.zeros((nstates, nobs))

    # initialize the filter
    st = np.zeros((nstates, 1)) # s_{t|t-1}
    pt = np.eye(nstates)*1000   # p_{t|t-1}

    s10[:,0] = np.squeeze(st)
    p10[:,:,0] = np.squeeze(pt)
 
    mt = Λ@st         # (nvar x 1)
    vt = Λ@pt@Λ.T + R # (nvar x nvar)
    ut = y[:,0] - mt  # (nvar x 1)

    lnl[0] = -0.5*(nvar*np.log(2*np.pi) + np.log(det(vt)) + ut.T@inv(vt)@ut)

    K  = pt@Λ.T@inv(vt)
    s0 = st + K@ut     # s_{t|t}
    p0 = pt - K@Λ@pt   # p_{t|t}

    s11[:,0]   = np.squeeze(s0)
    p11[:,:,0] = np.squeeze(p0)

    # recursion of filter
    for t in range(1,nobs):

        # propagation step
        st = Φ@s0          # (nstate x 1)
        pt = Φ@p0@Φ.T + Q  # (nstate x nstate)

        s10[:,t] = np.squeeze(st)
        p10[:,:,t] = np.squeeze(pt)

        # prediction step
        mt = Λ@st         # (nvar x 1)
        vt = Λ@pt@Λ.T + R # (nvar x nvar)
        ut = y[:,t] - mt  # (nvar x 1)

        lnl[t] = -0.5*(nvar*np.log(2*np.pi) + np.log(det(vt)) + ut.T@inv(vt)@ut)

        # update step
        K  = pt@Λ.T@inv(vt)
        s0 = st + K@ut     # s_{t|t}
        p0 = pt - K@Λ@pt   # p_{t|t}

        # save the filtered states
        s11[:,t]   = np.squeeze(s0)
        p11[:,:,t] = np.squeeze(p0)

    # recursion of smooth
    
    ss[:,nobs-1] = np.squeeze(s0)

    for i in range(1,nobs):
        
        t = nobs - i - 1
        
        Jt      = p11[:,:,t] @ Φ.T @ inv(p10[:,:,t])
        ss[:,t] = s11[:,t] + Jt@(ss[:,t+1] - s10[:,t+1]) 

    return (s11, ss, lnl)


#------------------------------------------
# negative of loglikilihood unconstrained
#------------------------------------------

def neglog(para, y):
    
    Λ, Φ, R, Q   = state_space(para)
    s11, ss, lnl = kalman(Λ, Φ, R, Q, y.T)
    
    return -sum(lnl)
#------------------------------------------
# negative of loglikilihood constrained
#------------------------------------------

def neglog_const(para, y):
    
    para[0] = 1
    
    Λ, Φ, R, Q   = state_space(para)
    s11, ss, lnl = kalman(Λ, Φ, R, Q, y.T)
    
    return -sum(lnl)
