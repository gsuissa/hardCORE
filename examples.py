import numpy as np
import hardcore

# Note that throughout, all masses and radii and given in Earth units.

print(' ')
print('=== hardCORE EXAMPLES! ===')
print(' ')

# Example 1: Calculating the radius of a planet from its mass and core
# radius fraction, under the assumption of an iron + silicate two-layer
# model.
# i.e. (M,CRF) -> (R)

M_test = 1.0
CRF_test = 0.5

print('EXAMPLE 1: CONVERTING (M,CRF) -> R')
print('Assuming M = ',M_test,' and CRF = ',CRF_test)
print('R = ',np.round(hardcore.forward(M_test,CRF_test),3))
print(' ')

# Example 2: Inverting a choice of mass and radius (M_test and R_test
# respectively) into a minimum core radius fraction, maximum core radius
# fraction and marginalized core radius fraction (CRFmin, CRFmax and CRFmarg
# respectively). Results are general, including four-layer planets.
# i.e. (M,R) -> (CRF)

M_test = 1.0
R_test = 1.0

print('EXAMPLE 2: CONVERTING (M,R) -> CRF')
print('Assuming M = ',M_test,' and R = ',R_test)
print('[ CRFmin CRFmax CRFmarg] = ',np.round(hardcore.invert(M_test,R_test),3))
print(' ')

# Example 3: Sensitivity analysis example. Choose a mass and radius and
# associated uncertainties to boostrap predicted uncertainties on
# CRF. Calculation assumes no covariance between mass and radius, and
# that each term is described by a normal (although feel free to edit
# below as desired!)

M_test = 1.0   # mass measurement
M_delta = 0.01 # error on mass
R_test = 1.0   # radius measurement
R_delta = 0.01 # error on radius

n = 10000 # number of trials to use for Monte Carlo
masses = [0 for i in range(n)]
radii = [0 for i in range(n)]
CRFmins = [0 for i in range(n)]
CRFmargs = [0 for i in range(n)]
CRFmaxs = [0 for i in range(n)]
for i in range(n):
  masses[i] = -1.0
  while masses[i] < 0:
    masses[i] = np.random.normal(M_test,M_delta)
  radii[i] = -1.0
  while radii[i] < 0:
    radii[i] = np.random.normal(R_test,R_delta)
  temp=hardcore.invert(masses[i],radii[i])
  CRFmins[i] = temp[0]
  CRFmaxs[i] = temp[1]
  CRFmargs[i] = temp[2]
 
print('EXAMPLE 3: CONVERTING (M+/-m,R+/-r) -> CRF+/-crf')
print('Assuming M = [',M_test,'+/-',M_delta,'] and R = (',R_test,'+/-',R_delta,')')
print('CRFmin = ',np.median(CRFmins),'+/-',np.std(CRFmins))
print('CRFmax = ',np.median(CRFmaxs),'+/-',np.std(CRFmaxs))
print('CRFmarg = ',np.median(CRFmargs),'+/-',np.std(CRFmargs))
print(' ')