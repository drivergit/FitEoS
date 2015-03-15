#!/usr/bin/env python
                                                                   
'''Fitting Vinet EOS to data'''

#import modules this way so function calls are explicit to each module. Avoids crashes caused by confusion
import pylab
import scipy
import scipy.optimize
import sympy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy.plotting import plot as pltsympy
from scipy.interpolate import interp1d
from sympy.polys.polyfuncs import interpolate
from sympy.abc import x

#Define functions; Main program is at the bottom

#########################################################################################
######## 4rd Order Polynomial fit to estimate intial Vinet fit parameters ###############
#########################################################################################
def poly4Fit(volumes,energies):
    a4,b4,c4,d4,e4 = pylab.polyfit(volumes,energies,4) #this is from pylab
    #a2,b2,c2 = pylab.polyfit(volumes,energies,2) #this is from pylab
    V = sympy.Symbol('V') #sympy variable for taking derivatives    
    E = a4*pow(V,4) + b4*pow(V,3) + c4*pow(V,2) + d4*V + e4 #4th order polynomial E(V)
    #E = a2*pow(V,2) + b2*V + c2 #4th order polynomial E(V)
    ELambdaFunction = sympy.lambdify(V,E) #make a function E(V)
    Eprime = sympy.diff(E,V) #take derivative of E wrt to V
    EprimeLambdaFunction = sympy.lambdify(V,Eprime) #function dE/dV of V
    
    #print "Eprimefn= ", EprimeLambdaFunction(V)

    #vfit = np.linspace(min(volumes),max(volumes),500)
    #plt.plot(volumes, energies)
    #plt.plot(vfit,ELambdaFunction(vfit))
    #plt.show()

    
    V0poly = scipy.optimize.brentq(EprimeLambdaFunction,min(volumes),max(volumes)) # find the root of dE/dV=0 with scipy's brentq method in interval range
    #print "V04thpoly(Bohr$^3$)=", V0poly
    E0poly=ELambdaFunction(V0poly) #evaluate E(V) at V=V0
    #print "E04thpoly(Ha)= ", E0poly
    #Pressure = -EprimeLambdaFunction   #pressure P = -dE/dV
    #print "Pressure= ", Pressure                                                                                                                   
    BulkModulus = V*sympy.diff(E,V,2)  #Bulk modulus B = -V(dP/dV) = +V(d^2E/dV^2)
    BLambdaFunction = sympy.lambdify(V,BulkModulus) #make a function for B(V)
    B0poly=BLambdaFunction(V0poly)*29421.010901602753 #evaluate B(V) at V=V0 and convert to GPa
    #print "B04thpoly(GPa)= ", B0poly
    
    #Bprime=BulkModulus.diff(P) #implement this estimate later
    BP0poly = 4.0 #4.0 is a generally good guess
    #print "BP0poly (picked as 4)= ", BP0poly

    return [V0poly, B0poly, BP0poly, E0poly, ELambdaFunction]


#########################################################################################
############################# Vinet fit: find Vinet parameters ##########################
#########################################################################################
def GetVinetParameters(volumes,energies,errors,V0poly,B0poly,BP0poly,E0poly):

    def VinetCF(vol,V0,B0,BP0,E0): #define a bit different for curve_fit becuase the call is different
        
        #Vinet eqn from PRB 73, 104303 (2006)
        xsi = (3.0/2.0)*(BP0-1)
        xx = pow( (vol/V0), 1.0/3.0)
        #    B0Ha=B0/29421.010901602753
        E = E0 + (9.0*B0*V0/(pow(xsi,2.0))) * (1.0 + (xsi*(1.0-xx)-1.0) * np.exp(xsi*(1.0-xx)))
        
        return E

    x0 = [V0poly, B0poly, BP0poly, E0poly] #initial parameter guesses (from 4th order polynomial fit)
    vinetparsCF, pcovCF = scipy.optimize.curve_fit(VinetCF, volumes, energies, p0=x0,sigma=errors) #curve_fit() from scipy

    #scale the covariance matrix to depend on errors
    chi = (energies - VinetCF(volumes, *vinetparsCF)) / errors
    chi2 = (chi**2).sum()
    dof = len(energies) - len(vinetparsCF)
    factor = (chi2 / dof)
    pcovCFscaled = pcovCF/factor

    #Final Vinet fit parameters
    V0vinet=vinetparsCF[0]
    B0vinet=vinetparsCF[1]
    BP0vinet=vinetparsCF[2]
    E0vinet=vinetparsCF[3]

    return [V0vinet,B0vinet,BP0vinet,E0vinet,pcovCFscaled]


#######################################################
#Compute Vinet energy function and propagate errors
#######################################################
def GetEnergyAndError(vfit,V0vinet,B0vinet,BP0vinet,E0vinet,pcovCFscaled):
    
    #Symbolic derivatives of Vinet Energy
    V0sym,B0sym,BP0sym,E0sym,Vsym=sympy.symbols('V0sym,B0sym,BP0sym,E0sym,Vsym')
    xsi = (3.0/2.0)*(BP0sym-1)
    xx = pow( (Vsym/V0sym), 1.0/3.0)
    #B0Ha=B0sym/29421.010901602753
    Energy=E0sym + (9.0*B0sym*V0sym/(pow(xsi,2.0))) * (1.0 + (xsi*(1.0-xx)-1.0)*sympy.exp(xsi*(1.0-xx)))
    DEV0sym = sympy.diff(Energy,V0sym)
    DEB0sym = sympy.diff(Energy,B0sym)
    DEBP0sym = sympy.diff(Energy,BP0sym)
    DEE0sym = sympy.diff(Energy,E0sym)
    DEparams=[DEV0sym,DEB0sym,DEBP0sym,DEE0sym]
    
    #compute  sigmaE(V)
    #i=0,1,2,3 (V0,B0,BP0,E0)
    varianceE=0.0
    for i in range(0, 3):
        for j in range(0, 3):
            varianceE+=DEparams[i]*DEparams[j]*pcovCFscaled[i][j]
            
    sigmaE=pow(varianceE,1.0/2.0)
    #sigmaE(V)
    sigmaEfn = sympy.lambdify((V0sym,B0sym,BP0sym,E0sym,Vsym),sigmaE) #function
    #print "sigmaE0=",sigmaEfn(V0vinet,B0vinet,BP0vinet,E0vinet,4200.0)

    Energyfn = sympy.lambdify((V0sym,B0sym,BP0sym,E0sym,Vsym),Energy) #function dE/dV of V
    npoints=len(vfit)
    Evals=np.zeros(npoints)
    EvalsPsigma=np.zeros(npoints)
    EvalsMsigma=np.zeros(npoints)
    for niter in range(0,npoints,1):
        Evals[niter]=Energyfn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])
        EvalsPsigma[niter]=Evals[niter]+sigmaEfn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])
        EvalsMsigma[niter]=Evals[niter]-sigmaEfn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])
        
    #np.seterr(all='print') #debug

    return [Energy,Evals,EvalsPsigma,EvalsMsigma]

#######################################################
#Compute pressure and propagate errors
#######################################################
def GetPressureAndError(Energy,vfit,V0vinet,B0vinet,BP0vinet,E0vinet,pcovCFscaled):

    V0sym,B0sym,BP0sym,E0sym,Vsym=sympy.symbols('V0sym,B0sym,BP0sym,E0sym,Vsym')
    Pressure = -sympy.diff(Energy,Vsym) #take derivative of E wrt to V
    DPV0sym = sympy.diff(Pressure,V0sym)
    DPB0sym = sympy.diff(Pressure,B0sym)
    DPBP0sym = sympy.diff(Pressure,BP0sym)
    DPE0sym = sympy.diff(Pressure,E0sym)
    DPparams=[DPV0sym,DPB0sym,DPBP0sym,DPE0sym]

    #compute  sigmaP(V)
    #i=0,1,2,3 (V0,B0,BP0,E0)
    varianceP=0.0
    for i in range(0, 3):
        for j in range(0, 3):
            varianceP+=DPparams[i]*DPparams[j]*pcovCFscaled[i][j]
            
    sigmaP=pow(varianceP,1.0/2.0)
    sigmaPfn = sympy.lambdify((V0sym,B0sym,BP0sym,E0sym,Vsym),sigmaP) #sigmaP(V)
    #print "sigmaP0=",sigmaPfn(V0vinet,B0vinet,BP0vinet,E0vinet,4200.0)*29421.010901602753 #Converted to GPa


    Pressurefn = sympy.lambdify((V0sym,B0sym,BP0sym,E0sym,Vsym),Pressure) #function dE/dV of V
    npoints=len(vfit)
    pvals=np.zeros(npoints)
    pvalsPsigma=np.zeros(npoints)
    pvalsMsigma=np.zeros(npoints)
    for niter in range(0,npoints,1):
        pvals[niter]=Pressurefn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])*29421.010901602753
        pvalsPsigma[niter]=pvals[niter]+sigmaPfn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])*29421.010901602753
        pvalsMsigma[niter]=pvals[niter]-sigmaPfn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])*29421.010901602753


    return [Pressure,pvals,pvalsPsigma,pvalsMsigma]


#######################################################
#Compute enthalpy and propagate errors
#######################################################
def GetEnthalpyAndError(Pressure,pvals,Energy,vfit,V0vinet,B0vinet,BP0vinet,E0vinet,pcovCFscaled):

    #Enthalpy

#    npoints=int(np.ceil((max(volumes)-min(volumes)))) #20 times the number ofsingle volumes steps
#    print "npoints= ",npoints
#    npoints=100
#    vfit = np.linspace(min(volumes),max(volumes),100)
#    vfit = np.linspace(min(volumes),max(volumes),npoints)

    V0sym,B0sym,BP0sym,E0sym,Vsym=sympy.symbols('V0sym,B0sym,BP0sym,E0sym,Vsym')
    Enthalpyfn=sympy.lambdify((V0sym,B0sym,BP0sym,E0sym,Vsym),Energy+Pressure*Vsym) #Enthalpy function in Ha
    #print "Enthalpyfn= ",Enthalpyfn(V0vinet,B0vinet,BP0vinet,E0vinet,4000.0)
    npoints=len(vfit)
    Hvals=np.zeros(npoints)
    for niter in range(0,npoints,1):
        Hvals[niter]=Enthalpyfn(V0vinet,B0vinet,BP0vinet,E0vinet,vfit[niter])

    #Fit a 10th order polynomial to the H(P) data to have an analytic form to solve for transition pressure and propagate error with MC
    P = sympy.Symbol('P') #sympy variable for taking derivatives
    Ha4,Hb4,Hc4,Hd4,He4,Hf4,Hg4,Hh4,Hi4,Hj4,Hk4 = pylab.polyfit(pvals,Hvals,10) #this is from pylab
    #Hpolyfn = Ha4*pow(P,4) + Hb4*pow(P,3) + Hc4*pow(P,2) + Hd4*P + He4 #4th order polynomial E(V)
    Hpolyfn = Ha4*pow(P,10) + Hb4*pow(P,9) + Hc4*pow(P,8) + Hd4*pow(P,7) + He4*pow(P,6)+ Hf4*pow(P,5) + Hg4*pow(P,4) + Hh4*pow(P,3) + Hi4*pow(P,2) + Hj4*P + Hk4
    HLambdaFunction = sympy.lambdify(P,Hpolyfn) #make a function E(V)
    HvalspolyL=np.zeros(npoints)
    for niter in range(0,npoints,1):
        HvalspolyL[niter]=HLambdaFunction(pvals[niter])
    
    #check how well the fit reproduces the data
    #diff= HvalspolyL-Hvals

    return [Hvals,Hpolyfn,HLambdaFunction]



def EoS1(vfit1,energies1,volumes1):
    #Get parameter estimates for Vinet EoS from a preliminary polynomial fit
    V0poly1, B0poly1, BP0poly1, E0poly1, ELambdaFunctionpoly1 = poly4Fit(volumes1,energies1)
    #Fit Vinet EoS to get V0,B0,B0',E0 parameters
    V0vinet1, B0vinet1, BP0vinet1, E0vinet1, pcovCFscaled1 = GetVinetParameters(volumes1,energies1,errors1,V0poly1,B0poly1,BP0poly1,E0poly1)
    #Compute list of energies and propagated errors
    Energy1, Evals1, EvalsPsigma1, EvalsMsigma1 =  GetEnergyAndError(vfit1,V0vinet1,B0vinet1,BP0vinet1,E0vinet1,pcovCFscaled1)
    #Compute list of pressures and propagated errors
    Pressure1, pvals1, pvalsPsigma1, pvalsMsigma1 = GetPressureAndError(Energy1,vfit1,V0vinet1,B0vinet1,BP0vinet1,E0vinet1,pcovCFscaled1)
    #Compute enthalpy function and propagated errors
    Hvals1, Hpolyfn1, HLambdaFunction1 = GetEnthalpyAndError(Pressure1,pvals1,Energy1,vfit1,V0vinet1,B0vinet1,BP0vinet1,E0vinet1,pcovCFscaled1)

    return [Evals1,EvalsPsigma1,EvalsMsigma1,pvals1,pvalsPsigma1,pvalsMsigma1,Hvals1,Hpolyfn1,HLambdaFunction1,V0vinet1,B0vinet1,BP0vinet1,E0vinet1,pcovCFscaled1]


def EoS2(vfit2,energies2,volumes2):
    #Get parameter estimates for Vinet EoS from a preliminary polynomial fit
    V0poly2, B0poly2, BP0poly2, E0poly2, ELambdaFunctionpoly2 = poly4Fit(volumes2,energies2)
    #Fit Vinet EoS to get V0,B0,B0',E0 parameters
    V0vinet2, B0vinet2, BP0vinet2, E0vinet2, pcovCFscaled2 = GetVinetParameters(volumes2,energies2,errors2,V0poly2,B0poly2,BP0poly2,E0poly2)
    #Compute list of energies and propagated errors
    Energy2, Evals2, EvalsPsigma2, EvalsMsigma2 =  GetEnergyAndError(vfit2,V0vinet2,B0vinet2,BP0vinet2,E0vinet2,pcovCFscaled2)
    #Compute list of pressures and propagated errors
    Pressure2, pvals2, pvalsPsigma2, pvalsMsigma2 = GetPressureAndError(Energy2,vfit2,V0vinet2,B0vinet2,BP0vinet2,E0vinet2,pcovCFscaled2)
    #Compute enthalpy function and propagated errors
    Hvals2, Hpolyfn2, HLambdaFunction2 = GetEnthalpyAndError(Pressure2,pvals2,Energy2,vfit2,V0vinet2,B0vinet2,BP0vinet2,E0vinet2,pcovCFscaled2)

    return [Evals2,EvalsPsigma2,EvalsMsigma2,pvals2,pvalsPsigma2,pvalsMsigma2,Hvals2,Hpolyfn2,HLambdaFunction2,V0vinet2,B0vinet2,BP0vinet2,E0vinet2,pcovCFscaled2]

def EoS3(vfit3,energies3,volumes3):
    #Get parameter estimates for Vinet EoS from a preliminary polynomial fit
    V0poly3, B0poly3, BP0poly3, E0poly3, ELambdaFunctionpoly3 = poly4Fit(volumes3,energies3)
    #Fit Vinet EoS to get V0,B0,B0',E0 parameters
    V0vinet3, B0vinet3, BP0vinet3, E0vinet3, pcovCFscaled3 = GetVinetParameters(volumes3,energies3,errors3,V0poly3,B0poly3,BP0poly3,E0poly3)
    #Compute list of energies and propagated errors
    Energy3, Evals3, EvalsPsigma3, EvalsMsigma3 =  GetEnergyAndError(vfit3,V0vinet3,B0vinet3,BP0vinet3,E0vinet3,pcovCFscaled3)
    #Compute list of pressures and propagated errors
    Pressure3, pvals3, pvalsPsigma3, pvalsMsigma3 = GetPressureAndError(Energy3,vfit3,V0vinet3,B0vinet3,BP0vinet3,E0vinet3,pcovCFscaled3)
    #Compute enthalpy function and propagated errors
    Hvals3, Hpolyfn3, HLambdaFunction3 = GetEnthalpyAndError(Pressure3,pvals3,Energy3,vfit3,V0vinet3,B0vinet3,BP0vinet3,E0vinet3,pcovCFscaled3)

    return [Evals3,EvalsPsigma3,EvalsMsigma3,pvals3,pvalsPsigma3,pvalsMsigma3,Hvals3,Hpolyfn3,HLambdaFunction3,V0vinet3,B0vinet3,BP0vinet3,E0vinet3,pcovCFscaled3]


def EoS4(vfit4,energies4,volumes4):
    #Get parameter estimates for Vinet EoS from a preliminary polynomial fit
    V0poly4, B0poly4, BP0poly4, E0poly4, ELambdaFunctionpoly4 = poly4Fit(volumes4,energies4)
    #Fit Vinet EoS to get V0,B0,B0',E0 parameters
    V0vinet4, B0vinet4, BP0vinet4, E0vinet4, pcovCFscaled4 = GetVinetParameters(volumes4,energies4,errors4,V0poly4,B0poly4,BP0poly4,E0poly4)
    #Compute list of energies and propagated errors
    Energy4, Evals4, EvalsPsigma4, EvalsMsigma4 =  GetEnergyAndError(vfit4,V0vinet4,B0vinet4,BP0vinet4,E0vinet4,pcovCFscaled4)
    #Compute list of pressures and propagated errors
    Pressure4, pvals4, pvalsPsigma4, pvalsMsigma4 = GetPressureAndError(Energy4,vfit4,V0vinet4,B0vinet4,BP0vinet4,E0vinet4,pcovCFscaled4)
    #Compute enthalpy function and propagated errors
    Hvals4, Hpolyfn4, HLambdaFunction4 = GetEnthalpyAndError(Pressure4,pvals4,Energy4,vfit4,V0vinet4,B0vinet4,BP0vinet4,E0vinet4,pcovCFscaled4)

    return [Evals4,EvalsPsigma4,EvalsMsigma4,pvals4,pvalsPsigma4,pvalsMsigma4,Hvals4,Hpolyfn4,HLambdaFunction4,V0vinet4,B0vinet4,BP0vinet4,E0vinet4,pcovCFscaled4]

#############
#MAIN PROGRAM
#############

#######
#QMC
#######
#read in data set #1
volumes1,energies1,errors1=np.loadtxt('HSdmc.dat',unpack=True)
#make a vector to evaluate fits on with a lot of points so it looks smooth
vfit1 = np.linspace(min(volumes1),max(volumes1),500)
#read in data set #2
volumes2,energies2,errors2=np.loadtxt('LSdmc.dat',unpack=True)
#make a vector to evaluate fits on with a lot of points so it looks smooth
vfit2 = np.linspace(min(volumes2),max(volumes2),500)

Evals1, EvalsPsigma1, EvalsMsigma1, pvals1, pvalsPsigma1, pvalsMsigma1, Hvals1, Hpolyfn1, HLambdaFunction1, V0vinet1, B0vinet1, BP0vinet1, E0vinet1, pcovCFscaled1 = EoS1(vfit1,energies1,volumes1)
print " "
print "HS DMC paramters:"
print "V0, sigma= ",V0vinet1,np.sqrt(pcovCFscaled1[0][0])
print "B0, sigma= ",B0vinet1*29421.010901602753,np.sqrt(pcovCFscaled1[1][1])*29421.010901602753
print "BP0, sigma= ",BP0vinet1,np.sqrt(pcovCFscaled1[2][2])
print "E0, sigma= ",E0vinet1,np.sqrt(pcovCFscaled1[3][3])
print " "

Evals2, EvalsPsigma2, EvalsMsigma2, pvals2, pvalsPsigma2, pvalsMsigma2, Hvals2, Hpolyfn2, HLambdaFunction2, V0vinet2, B0vinet2, BP0vinet2, E0vinet2, pcovCFscaled2 = EoS2(vfit2,energies2,volumes2)
print " "
print "LS DMC paramters:"
print "V0, sigma= ",V0vinet2,np.sqrt(pcovCFscaled2[0][0])
print "B0, sigma= ",B0vinet2*29421.010901602753,np.sqrt(pcovCFscaled2[1][1])*29421.010901602753
print "BP0, sigma= ",BP0vinet2,np.sqrt(pcovCFscaled2[2][2])
print "E0, sigma= ",E0vinet2,np.sqrt(pcovCFscaled2[3][3])


#Use Monte Carlo method to compute the average transition pressure and its error bar
print " "
P = sympy.Symbol('P') #sympy variable
nMCiterations = int(80)
energies1rand=np.zeros(len(energies1))
energies2rand=np.zeros(len(energies2))
TransitionPressureMC=np.zeros(nMCiterations)
for i in range(0, nMCiterations):
    print "Determining error in Transition Pressure: MC iteration = ",i
    for j in range(0,len(energies1)):
        energies1rand[j]=np.random.normal(energies1[j], errors1[j], 1) #generate random Normal fulcuations within the simga of the energies1
    for k in range(0,len(energies2)):
        energies2rand[k]=np.random.normal(energies2[k], errors2[k], 1) #generate random Normal fulcuations within the simga of the energies2
    #print "energies= ",energies1,energies2
    Evals1MC,EvalsPsigma1MC,EvalsMsigma1MC,pvals1MC,pvalsPsigma1MC,pvalsMsigma1MC,Hvals1MC,Hpolyfn1MC,HLambdaFunction1MC,V0vinet1MC,B0vinet1MC,BP0vinet1MC,E0vinet1MC,pcovCFscaled1MC = EoS1(vfit1,energies1rand,volumes1)
    Evals2MC,EvalsPsigma2MC,EvalsMsigma2Mc,pvals2MC,pvalsPsigma2MC,pvalsMsigma2MC,Hvals2MC,Hpolyfn2MC,HLambdaFunction2MC,V0vinet2MC,B0vinet2MC,BP0vinet2MC,E0vinet2MC,pcovCFscaled2MC = EoS2(vfit2,energies2rand,volumes2)
    Hpolyfn1m2 = Hpolyfn1MC-Hpolyfn2MC
    HLambdaFunction3MC = sympy.lambdify(P,Hpolyfn1m2) #make a function E(V)
    TransitionPressureMC[i] = scipy.optimize.brentq(HLambdaFunction3MC,min(pvals1),max(pvals1))

#Transition Pressure and error bar
TransP = np.mean(TransitionPressureMC)
Psigma = np.std(TransitionPressureMC)
print "DMC HS-LS Transition Pressure= ",TransitionPressureMC
print "TransP mean= ",TransP
print "TransP Standard Deviation= ",Psigma


#######
#DFT
#######
#read in data set #1
volumes3,energies3,errors3=np.loadtxt('HSdft.dat',unpack=True)
#make a vector to evaluate fits on with a lot of points so it looks smooth
vfit3 = np.linspace(min(volumes1),max(volumes1),500)
#read in data set #2
volumes4,energies4,errors4=np.loadtxt('LSdft.dat',unpack=True)
#make a vector to evaluate fits on with a lot of points so it looks smooth
vfit4 = np.linspace(min(volumes2),max(volumes2),500)

Evals3, EvalsPsigma3, EvalsMsigma3, pvals3, pvalsPsigma3, pvalsMsigma3, Hvals3, Hpolyfn3, HLambdaFunction3, V0vinet3, B0vinet3, BP0vinet3, E0vinet3, pcovCFscaled3 = EoS3(vfit3,energies3,volumes3)
print " "
print "HS DFT paramters:"
print "V0, sigma= ",V0vinet3,np.sqrt(pcovCFscaled3[0][0])
print "B0, sigma= ",B0vinet3*29421.010901602753,np.sqrt(pcovCFscaled3[1][1])*29421.010901602753
print "BP0, sigma= ",BP0vinet3,np.sqrt(pcovCFscaled3[2][2])
print "E0, sigma= ",E0vinet3,np.sqrt(pcovCFscaled3[3][3])
print " "

Evals4, EvalsPsigma4, EvalsMsigma4, pvals4, pvalsPsigma4, pvalsMsigma4, Hvals4, Hpolyfn4, HLambdaFunction4, V0vinet4, B0vinet4, BP0vinet4, E0vinet4, pcovCFscaled4 = EoS4(vfit4,energies4,volumes4)
print " "
print "LS DFT paramters:"
print "V0, sigma= ",V0vinet4,np.sqrt(pcovCFscaled4[0][0])
print "B0, sigma= ",B0vinet4*29421.010901602753,np.sqrt(pcovCFscaled4[1][1])*29421.010901602753
print "BP0, sigma= ",BP0vinet4,np.sqrt(pcovCFscaled4[2][2])
print "E0, sigma= ",E0vinet4,np.sqrt(pcovCFscaled4[3][3])

#Use Monte Carlo method to compute the average transition pressure and its error bar
P = sympy.Symbol('P') #sympy variable
Hpolyfn3m4 = Hpolyfn3-Hpolyfn4
HLambdaFunction5 = sympy.lambdify(P,Hpolyfn3m4) #make a function E(V)
TransitionPressure = scipy.optimize.brentq(HLambdaFunction5,min(pvals3),max(pvals3))
print " "
print "DFT HS-LS Transition Pressure= ",TransitionPressure


#######
#plots
#######
mpl.rcParams.update({'font.size': 12})
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.subplot.bottom'] = 0.10    # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.top'] = 0.95    # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.right'] = 0.95    # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.left'] = 0.10    # the bottom of the subplots of the figure
mpl.rcParams['figure.subplot.wspace'] = 0.25
mpl.rcParams['figure.subplot.hspace'] = 0.25
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.handlelength'] = 2.5


#fig=plt.subplot(2,2,1)
#majorLocator   = MultipleLocator(5)
#minorLocator   = MultipleLocator(1)
#fig.yaxis.set_major_locator(majorLocator)
#fig.yaxis.set_minor_locator(minorLocator)
#majorLocator   = MultipleLocator(0.2)
#minorLocator   = MultipleLocator(0.05)
#fig.xaxis.set_major_locator(majorLocator)
#fig.xaxis.set_minor_locator(minorLocator)


#[x-E0vinet1 for x in energies1]
energies1=energies1-E0vinet1
energies2=energies2-E0vinet1
energies3=energies3-E0vinet3
energies4=energies4-E0vinet3
EvalsPsigma1=EvalsPsigma1-E0vinet1
EvalsMsigma1=EvalsMsigma1-E0vinet1
EvalsPsigma2=EvalsPsigma2-E0vinet1
EvalsMsigma2=EvalsMsigma2-E0vinet1
Evals3=Evals3-E0vinet3
Evals4=Evals4-E0vinet3

#print "energies1= ",energies1
#print "energies1shifted= ",energies1shifted
#print "EvalsPsigma1= ",EvalsPsigma1

#E(V) plot
plt.plot(volumes1*0.529177208**3/8,energies1,'ro',color='blue',markersize=4)
plt.plot(vfit1*0.529177208**3/8, EvalsPsigma1,color='blue',label='HS DMC')
plt.plot(vfit1*0.529177208**3/8, EvalsMsigma1,color='blue')
plt.fill_between(vfit1*0.529177208**3/8, EvalsMsigma1, EvalsPsigma1, color='blue')

plt.plot(volumes2*0.529177208**3/8,energies2,'ro',color='red',markersize=4)
plt.plot(vfit2*0.529177208**3/8, EvalsPsigma2,color='red',label='LS DMC')
plt.plot(vfit2*0.529177208**3/8, EvalsMsigma2,color='red')
plt.fill_between(vfit2*0.529177208**3/8, EvalsMsigma2, EvalsPsigma2, color='red')

plt.plot(volumes3*0.529177208**3/8,energies3,'ro',color='blue',markersize=4)
plt.plot(volumes4*0.529177208**3/8,energies4,'ro',color='red',markersize=4)
plt.plot(vfit3*0.529177208**3/8, Evals3,'--',color='blue',label='HS DFT')
plt.plot(vfit4*0.529177208**3/8, Evals4,'--',color='red',label='LS DFT')

#plt.xlim(45,90)
#plt.ylim(-0.25,3.0)
plt.xlabel('Volume ($\AA^3$)')
plt.ylabel('Energy (Ha)')
plt.legend(loc='best')
plt.show()

#P(V) plot
volumesExp,pressuresExp,=np.loadtxt('speziale_Fe17percent_2007_xAng_yGPa.dat',unpack=True)

plt.plot(volumesExp,pressuresExp,'ro',color='black',markersize=4)
plt.plot(vfit1*0.529177208**3/8, pvalsPsigma1,color='blue',label='HS DMC')
plt.plot(vfit1*0.529177208**3/8, pvalsMsigma1,color='blue')
plt.fill_between(vfit1*0.529177208**3/8, pvalsMsigma1, pvalsPsigma1, color='blue')

plt.plot(vfit2*0.529177208**3/8, pvalsPsigma2,color='red',label='LS DMC')
plt.plot(vfit2*0.529177208**3/8, pvalsMsigma2,color='red')
plt.fill_between(vfit2*0.529177208**3/8, pvalsMsigma2, pvalsPsigma2, color='red')

plt.plot(vfit3*0.529177208**3/8, pvals3,'--',color='blue',label='HS DFT')
plt.plot(vfit4*0.529177208**3/8, pvals4,'--',color='red',label='LS DFT')


plt.xlim(50,75)
plt.ylim(0,145)
plt.xlabel('Volume ($\AA^3$)')
plt.ylabel('Pressure (GPa)')
plt.legend(loc='best')
plt.show()

#Enthalpy H(P) plot
plt.plot(pvals1, Hvals1, label='H(V)')
plt.plot(pvals1, HLambdaFunction1(pvals1),label='10th order polynomial fit')
plt.plot(pvals2, Hvals2, label='H(V)')
plt.plot(pvals2, HLambdaFunction2(pvals2),label='10th order polynomial fit')
#plt.show()


