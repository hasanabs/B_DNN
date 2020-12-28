"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import numpy as np
import os
import time
import datetime
import fungsi_dnn as fn
from tensorflow import train
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


#########################################################################################
####################-Change specification inside this area-##############################

#Spec GSM
Nt, Np, Nr, M = 2, 2, 2, 4 #Transmit_antena, Active_antena, Receive_antena, Constelation(2,4,16)
Nt=Np #Since training focused on active antenna only so let Nt=Np
Ns=20000000 #Number of training data
SNR_Min,step,SNR_Max=10,4,10 #Not important at training process
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
# Es=Np #Not important at training process
modulation= fn.modulation(M)
bit_per_sym=np.int(np.sqrt(M))
keterangan=str(Np)+'.'+str(Np)+'.'+str(Nr)+'.'+str(M)+'_'+str(Ns)+'x'
    
#TAC Initialization
prob_TAC=fn.nchoosek(np.arange(1,Nt+1),Np)
tac=fn.optimum_TAC(prob_TAC,Nt,Np,N)

#Main Program
print("Nt."+str(Nt)+" Np."+str(Np)+" Nr."+str(Nr)+" M."+str(M))
Range=np.arange(SNR_Min,SNR_Max+1,step);
index_error=0
data=np.zeros((L,Ns*len(Range)),dtype=np.int)
# noise=np.zeros((Nr, SNR_Max+1), dtype=np.float16)
D=np.zeros(((2*Nr+2*Nr*Np),Ns*len(Range)),dtype=np.float16) #Input of NN
for SNR_dB in Range:  
    #Sending data   
    for send in range(Ns):
        data[:,send+index_error*Ns]=np.random.randint(2, size=L)
        x=fn.encode(data[:,send],Nt,Np,M,L1,tac,modulation)   
        # noise=fn.noise(SNR_dB, Nr, Es)
        H=fn.H(Nr, Nt)
        y=np.matmul(H,x) #+ noise
        H_reshape=H.reshape(-1,1)
        D[0:2*Nr,[send+index_error*Ns]]= np.concatenate((np.real(y),np.imag(y))) #Extraction for input of NN
        D[2*Nr:2*(Nr+Nr*Np),[send+index_error*Ns]]= np.concatenate((np.real(H_reshape),np.imag(H_reshape))) #Extraction for input of NN
    index_error=index_error+1

#Training and save    
start = time.time()
model=fn.decoder_symbol(2*Nr+2*Nr*Np, M, dnn_net, Np)
checkpoint_path="trainedNetworks/check_point_Np."+str(Np)+"_Nr."+str(Nr)+"_M."+str(M) +"/cp-{epoch:04d}.ckpt"
checkpoint_dir=os.path.dirname(checkpoint_path)
callbacks_list=ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_best_only=True)
log_dir = os.path.join("logs",   datetime.datetime.now().strftime("%Y-%m-%d_"))+keterangan
tensorboard = TensorBoard(log_dir, histogram_freq = 1, profile_batch=10)
latest=train.latest_checkpoint(checkpoint_dir)
train_labels_app=[]
train_labels=np.zeros((Ns*len(Range),1+Np))
for i in range(Np):
    for j in range(Ns*len(Range)): 
        train_labels[j,1+i:1+i+1]=fn.bi2de(data[L1+i*bit_per_sym:L1+(i+1)*bit_per_sym,j])
    train_labels_app.append(train_labels[:,1+i:1+i+1])
if latest!=None: #If previously there is a trained model with the same configuration will automatically continue the learning process
    model.load_weights(latest) 
model.fit(np.transpose(D),train_labels_app, validation_split=0.25, batch_size=512, epochs=50, callbacks=[callbacks_list, tensorboard], shuffle=True)

print ("\nTraining time : ", time.time()-start, "seconds.")  
