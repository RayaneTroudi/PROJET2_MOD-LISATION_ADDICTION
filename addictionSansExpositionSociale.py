###########################################                                        s
# _____________ BIBLIOTHEQUES ___________ #
###########################################
import numpy as np
import time as time
import matplotlib.pyplot as plt



##########################################                                        s
#_____________ CONSTANTES ______________ #
##########################################
q = 0.8
p = 0.6
alpha = 0.2
b = 0.5 #pas mettre la formuke (2*alpha) / q

###########################################                                        s
# _____________ CONDITIONS INITIALES ______________ #
###########################################
C0 = 2 #désir
E0 = 1
Sm = 1
S0 = Sm #self controle
h = 0.5
k = 0.375 #pareil pas de formule(p/q) * Sm
Em=0.01
# ##########################################                                        s
# #_____________ CONSTANTES ______________ #
# ##########################################
# q = 0.8
# p = 0.2
# alpha = 0.2
# gamma = 0.2
# b = (2*alpha) / q

# ###########################################                                        s
# # _____________ CONDITIONS INITIALES ______________ #
# ###########################################
# C0 = 2
# E0 = 1
# Sm = 0.2
# S0 = Sm
# h = p * Sm
# k = (p/q) * Sm

def Phi(C_t:float,S_t:float,E_t:float)->float:
    
    """Fonction modélisant l'état psychlogique

    Args:
        C (float): intensité de fringale ou de désir
        S (float): intensité de self-contrôle
        E (float): influence extérieur

    Returns:
        Phi (float) : état psychlogique
    """
    
    return C_t-S_t-E_t

def A(V_t:float,q:float)->float:
    """Fonction modélisant le passage à l'acte

    Args:
        V_t (float): se réferrer à V
        q (cste_float): voir zone constante 

    Returns:
        A_t (float): passage à l'acte
    """
    return V_t * q

def V(Phi_t:float)->float:
    """Fonction modélisant l'état addictif

    Args:
        P_t (float): se réferrer à la fonction Phi

    Returns:
        V_t (float): état addictif (0 = pas addict , float sinon)
    """
    return np.minimum(1,np.maximum(Phi_t,0))

def C(C_t : float ,A_t : float ,alpha : float,gamma : float) -> float:
    """Fonction modélisant l'intensité de fringale ou de désir

    Args:
        C_t (float): 
        A_t (_type_): _description_
        alpha (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    return C_t-alpha*C_t+gamma*A_t


def S(S_t:float,C_t:float,A_t:float,p:float,h:float,k:float,Smax:float)->float:
    """Fonction modélisant l'intensité de self-contrôle

    Args:
        S_t (float): fonction au pas précédent
        C_t (float): intensité de fringale
        A_t (float): passage à l'acte
        p (float): résiliance psychlologique
        h (float): compétition entre S et C
        k (float): coefficient de passage à l'acte
        Smax (float): maximum de S

    Returns:
        S_t1 (float) : self-contrôle au pas suivant
    """
    return S_t + p * np.maximum(0,Smax-S_t) - h*C_t - k*A_t



weeks = 100

ens_Phi = np.zeros(weeks)
ens_C = np.zeros(weeks+1)
ens_S = np.zeros(weeks+1)
ens_E = np.zeros(weeks+1)
ens_A = np.zeros(weeks)
ens_V = np.zeros(weeks)

ens_C[0] = C0
ens_S[0] = S0
ens_E[0] = E0



for w in range(1,weeks+1):
    
    ens_E[w]=ens_E[w-1]-Em
    
    gamma=b*np.minimum(1,1-ens_C[w-1])
    
    ens_Phi[w-1] = Phi(ens_C[w-1],ens_S[w-1],ens_E[w-1])
    
    ens_V[w-1] = V(ens_Phi[w-1])
    
    ens_A[w-1] = A(ens_V[w-1],q)
    
    ens_C[w] = C(ens_C[w-1],ens_A[w-1],alpha,gamma)
    
    ens_S[w] = S(ens_S[w-1],ens_C[w-1],ens_A[w-1],p,h,k,max(ens_S[0:w]))

print(ens_Phi)
    
plt.plot(np.arange(0,weeks+1,1),ens_S,label="Self-control")
plt.plot(np.arange(0,weeks+1,1),ens_C,label="Fringale")

# plt.plot(np.arange(0,weeks,1),ens_Phi,label="phi")

plt.xlabel("Temps en semaines")
plt.ylabel("")
print(np.shape(ens_A))
plt.legend()
plt.grid()
plt.show()





###########################################                                    
# _________________MAIN__________________ #
###########################################
print(ens_V)





