"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import os
import time
import numpy as np
import fungsi_dnn as fn
import matplotlib.pyplot as plt

#########################################################################################
####################-Change specification inside this area-##############################

#Spec GSM
Nt, Np, Nr, M = 4, 2, 2, 4 #Transmit_antena, Active_antena, Receive_antena, Constelation(2,4,16)
Ns=100000 #Number of transmit time slot
SNR_Min,step,SNR_Max=0,4,24 

####################-Change specification inside this area-##############################
#########################################################################################


#Basic Properties of SM
L1= np.int(np.floor(np.log2(fn.nck(Nt,Np))))
L2=np.int(Np*np.log2(M))
L=L1+L2
N= np.int(np.power(2,L1))
Transmit_bit=L*Ns
Es=Np
modulation= fn.modulation(M)

#TAC Initialization
prob_TAC=fn.nchoosek(np.arange(1,Nt+1),Np)
tac=fn.optimum_TAC(prob_TAC,Nt,Np,N)

#Generate check bit ML
x_compare=np.zeros((Nt, np.power(2,L)), dtype=np.complex128)
x1=np.zeros((np.power(2,L),L), dtype=np.int)
for i in range(np.power(2,L)):
    x1[i,:]=fn.de2bi(i,L)
    x_compare[:,[i]]=fn.encode(x1[i,:],Nt,Np,M,L1,tac,modulation)

#Main Program
print("Nt."+str(Nt)+" Np."+str(Np)+" Nr."+str(Nr)+" M."+str(M))
Range=np.arange(SNR_Min,SNR_Max+1,step);
Error_bit=np.zeros((len(Range), 1), dtype=np.float16)
index_error=0
predict_stop=0
for SNR_dB in Range:
    Err_acumulation=0;    
    #Sending data   
    for send in range(Ns):
        data=np.random.randint(2, size=L)
        x=fn.encode(data,Nt,Np,M,L1,tac,modulation)
        noise=fn.noise(SNR_dB, Nr,Es)
        H=fn.H(Nr, Nt)
        y=np.matmul(H,x)+noise
        
        #ML Detector
        predict_start = time.time()
        min_ML=500;
        index_ML=0;
        for cek in range(np.power(2,L)):
            compare=np.linalg.norm(y-np.matmul(H,x_compare[:,[cek]]), 'fro')
            if min_ML>compare:
                min_ML=compare
                index_ML=cek
        predict_stop = predict_stop + time.time()-predict_start
        #ML Detector
        
        Err_acumulation+=(data!=x1[index_ML,:]).sum()

    Error_bit[index_error,0]=Err_acumulation/Transmit_bit
    print("SNR="+str(SNR_dB)+" n_Error="+str(Err_acumulation)+" transmited="+str(Transmit_bit)+" BER="+str(Error_bit[index_error,0]))
    index_error=index_error+1

#Plot
plt.plot(Range, Error_bit, 'ro-', linewidth=1, label="ML")
plt.legend(loc='upper right', fontsize='x-large')
plt.axis([SNR_Min, SNR_Max, np.min(Error_bit[np.nonzero(Error_bit)]), 1e-0])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.minorticks_on()
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor',alpha=0.4)
plt.suptitle('GSM-MIMO', fontsize='x-large', fontweight='bold')
plt.title('Nt='+str(Nt)+' Np='+str(Np)+' Nr='+str(Nr)+' M='+str(M)+' Ns='+str(Ns), fontsize='large', fontweight='book')
plt.show()   
if not os.path.exists('results'): os.makedirs('results')
plt.savefig('results/ML_Nt.'+str(Nt)+'_Np.'+str(Np)+'_Nr.'+str(Nr)+'_M.'+str(M)+'_Ns.'+str(Ns)+'.png')
print ("Time Complexity: ", predict_stop, "seconds.")
