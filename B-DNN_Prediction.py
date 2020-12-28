"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import os
import time
import numpy as np
import fungsi_dnn as fn
from tensorflow import train
import matplotlib.pyplot as plt


#########################################################################################
####################-Change specification inside this area-##############################

#Spec GSM
Nt, Np, Nr, M = 4, 2, 2, 4 #Transmit_antena, Active_antena, Receive_antena, Constelation(2,4,16)
Ns=100000 #Number of transmit time slot
SNR_Min,step,SNR_Max=0,4,24 

if M==2:
    dnn_net=np.array([128,64,32])
elif M==4:
    dnn_net=np.array([256,128,64])
else:
    dnn_net=np.array([512,256,128])
#dnn_net is number of node for each layer, and number of layer will follow number of array element

####################-Change specification inside this area-##############################
#########################################################################################
        

#Basic Properties of GSM
L1= np.int(np.floor(np.log2(fn.nck(Nt,Np))))
L2=np.int(Np*np.log2(M))
L=L1+L2
N= np.int(np.power(2,L1))
Transmit_bit=L*Ns
Es=Np
modulation= fn.modulation(M)
bit_per_sym=np.int(np.sqrt(M))

#TAC Initialization
prob_TAC=fn.nchoosek(np.arange(1,Nt+1),Np)
tac=fn.optimum_TAC(prob_TAC,Nt,Np,N)

#Build the model
model=fn.decoder_symbol(2*Nr+2*Nr*Np, M, dnn_net, Np)
checkpoint_path="trainedNetworks/check_point_Np."+str(Np)+"_Nr."+str(Nr)+"_M."+str(M) +"/cp-{epoch:04d}.ckpt"
checkpoint_dir=os.path.dirname(checkpoint_path)
latest=train.latest_checkpoint(checkpoint_dir)
if latest==None: #If previously there is a trained model with the same configuration will automatically continue the learning process
    print("\nThe model with specification Np="+str(Np)+" Nr="+str(Nr)+" M="+str(M)+" is not available yet, you might run B-DNN_Training.py with that specification before running B-DNN_Prediction.py file")
    raise SystemExit
model.load_weights(latest) 

#Main Program
print("Nt."+str(Nt)+" Np."+str(Np)+" Nr."+str(Nr)+" M."+str(M))
Range=np.arange(SNR_Min,SNR_Max+1,step);
Error_bit=np.zeros((len(Range), 1), dtype=np.float16)
D=np.zeros((2*Nr+2*Nr*Np,Ns*N),dtype=np.float16)
noise=np.zeros((Nr, SNR_Max+1), dtype=np.float16)
index_error=0
predict_stop=0
for SNR_dB in Range:    
    data=np.zeros((L,Ns),dtype=np.int)
    y=np.zeros((Nr,Ns),dtype=np.complex)
    H_collect=np.zeros((Ns*N,Nr,Np),dtype=np.complex)
    b_head=np.zeros((Ns*N,L),dtype=np.int)
    s_head=np.zeros((Ns*N,Np,1),dtype=np.complex64)
    
    #Sending data  
    for send in range(Ns):
        data[:,send]=np.random.randint(2, size=L)
        x=fn.encode(data[:,[send]],Nt,Np,M,L1,tac,modulation)   
        noise=fn.noise(SNR_dB, Nr, Es)
        H=fn.H(Nr, Nt)
        y[:,[send]]=np.matmul(H,x) + noise 
        
        for i in range(N): #Considered using ASIC-based to split
            H_active=H[:,tac[i,:]-1]
            H_act_reshape=H_active.reshape(-1,1)
            D[0:2*Nr,[send+Ns*i]]= np.concatenate((np.real(y[:,[send]]),np.imag(y[:,[send]])))
            D[2*Nr:2*(Nr+Nr*Np),[send+Ns*i]]= np.concatenate((np.real(H_act_reshape),np.imag(H_act_reshape)))
            b_head[send+Ns*i,0:L1]=fn.de2bi(i,L1)
            H_collect[send+Ns*i,:,:]=H_active
        
    #Predict the signal
    predict_start = time.time() 
    predictions = model.predict(np.transpose(D))
    rounded_predictions = np.argmax(predictions,axis=2)
    for send in range(Ns):
        for i in range(N):
            s_head[send+Ns*i,:,0]=modulation[rounded_predictions[:,send+Ns*i]]
            for j in range(Np):
                b_head[send+Ns*i,L1+j*bit_per_sym:L1+(j+1)*bit_per_sym]=fn.de2bi(rounded_predictions[j,send+Ns*i],bit_per_sym)
    H_s_head=np.matmul(H_collect,s_head) #Size is (Ns*N,Nr,1)
    H_s_head=np.reshape(H_s_head,(-1,H_s_head.shape[2]*N,H_s_head.shape[1]),order='F').transpose(0,2,1) #Size become (Ns,Nr,N)
    index_min=np.argmin(np.linalg.norm(np.reshape(y.T,(-1,Nr,1))-H_s_head,axis=1),axis=1) 
    decoded=b_head[Ns*index_min+np.arange(Ns),:]
    predict_stop = predict_stop + time.time()-predict_start        
    #Predict the signal
    
    #Calculate and print the BER for each SNR
    err_acumulation=(data!=np.transpose(decoded)).sum()
    Error_bit[index_error,0]=err_acumulation/Transmit_bit
    print("SNR="+str(SNR_dB)+" n_Error="+str(err_acumulation)+" transmited="+str(np.int(Transmit_bit))+" BER="+str(Error_bit[index_error,0]))
    index_error=index_error+1

#Plot
plt.plot(Range, Error_bit, 'g*-', linewidth=1, label="B-DNN")
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
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=1)
plt.text(1, 0.6, "Time complexity= "+str(round(predict_stop,2))+" seconds", verticalalignment="bottom", rotation=0, size=8, bbox=bbox_props)
plt.show()   
if not os.path.exists('../results'): os.makedirs('../results')
plt.savefig('../results/B-DNN_Nt.'+str(Nt)+'_Np.'+str(Np)+'_Nr.'+str(Nr)+'_M.'+str(M)+'_Ns.'+str(Ns)+'.png')
print ("Time Complexity: ", predict_stop, "seconds.")
