import numpy as np
import scipy.stats as ss

'''
計算implied volatility(Corrado su)_買權
'''
def implied_call(im):
    v=im[0]
    sk=im[1]
    ku=im[2]
    d1=(np.log(F/K)+(r+v**2/2)*T)/(v*np.sqrt(T))
    d2=(np.log(F/K)+(r-v**2/2)*T)/(v*np.sqrt(T))
    nd1=(1/np.sqrt(2*np.pi))*np.exp(-(1^2)/2)
    Q3=(1/(3*2))*(F*v*np.sqrt(T))*((2*v*np.sqrt(T)-d1)*nd1-v**2*T*ss.norm.cdf(d1))
    Q4=(1/(4*3*2))*(F*v*np.sqrt(T))*((d1**2-1-3*v*np.sqrt(T)*(d1-v*np.sqrt(T)))*nd1+v**3*T**(3/2)*ss.norm.cdf(d1)) 
    call=(np.exp(-r*T)*((F*ss.norm.cdf(d1))-(K*ss.norm.cdf(d2))))+sk*Q3+ku*Q4
    #call=(S0*ss.norm.cdf(d1)-K*np.exp(-r*TT)*ss.norm.cdf(d2))+sk*Q3+ku*Q4
    return ((C-call)**2)
	

'''
計算implied volatility(Corrado su)_賣權
'''
def implied_put(im):
    v=im[0]
    sk=im[1]
    ku=im[2]
    d1=(np.log(F/K)+(r+v**2/2)*T)/(v*np.sqrt(T))
    d2=(np.log(F/K)+(r-v**2/2)*T)/(v*np.sqrt(T))
    nd1=(1/np.sqrt(2*np.pi))*np.exp(-(1^2)/2)
    Q3=(1/(3*2))*(F*v*np.sqrt(T))*((2*v*np.sqrt(T)-d1)*nd1-v**2*T*ss.norm.cdf(d1))
    Q4=(1/(4*3*2))*(F*v*np.sqrt(T))*((d1**2-1-3*v*np.sqrt(T)*(d1-v*np.sqrt(T)))*nd1+v**3*T**(3/2)*ss.norm.cdf(d1)) 
    put=(np.exp(-r*T)*((K*ss.norm.cdf(d2))-(F*ss.norm.cdf(d1))))+sk*Q3+ku*Q4
    #call=(S0*ss.norm.cdf(d1)-K*np.exp(-r*TT)*ss.norm.cdf(d2))+sk*Q3+ku*Q4
    return ((P-put)**2)
