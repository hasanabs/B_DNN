"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import numpy as np
import itertools
from tensorflow import keras
from tensorflow.keras.layers import  Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD

def nck(n,k):
    return np.math.factorial(n)/np.math.factorial(k)/np.math.factorial(n-k)

def nchoosek(arr, k):
    return np.array(list(itertools.combinations(arr, k)))

def optimum_TAC(all_TAC, n, r, size_comb):
    ukuran=np.zeros(n,dtype=int)
    while(len(all_TAC)>size_comb):
        for i in range(n):
            ukuran[i]=(all_TAC==i+1).sum()
        idx_rem=0;
        remaining_idx=np.arange(len(all_TAC))
        sort_remove=np.argsort(-ukuran)
        while(len(remaining_idx)>1):
            old_remaining_idx=remaining_idx
            remaining_idx=remaining_idx[np.where((all_TAC[remaining_idx,:]==sort_remove[idx_rem]+1))[0]]
            if (len(remaining_idx)==0):
                idx=0
                while(len(remaining_idx)==0):
                   remaining_idx=old_remaining_idx[np.where((all_TAC[old_remaining_idx,:]==sort_remove[idx]+1))[0]]
                   idx+=1
            idx_rem+=1
        all_TAC=np.delete(all_TAC, (remaining_idx), axis=0)
    return all_TAC

def bi2de(arr):
    result=0
    for i in range(len(arr)):result+=np.power(2,i)*arr[len(arr)-1-i]
    return result

def de2bi(decimal, L_bit):
    arr=np.zeros((1,L_bit), dtype=np.int8)
    for i in range(L_bit): 
        arr[0,(L_bit-i-1)]=decimal%2
        decimal=decimal>>1
    return arr

def encode(dat, Nt, Np, M, L1, tac, modulation):
    x=np.zeros((Nt,1), dtype=np.complex64)
    for i in range(Np):x[tac[bi2de(dat[0:L1]),i]-1,0]=modulation[bi2de(dat[L1+np.int(np.log2(M))*(i):np.int(np.log2(M))*(i+1)+L1])]
    return x

def modulation(M):
    if M==2: modulation=np.array([-1+0j, 1+0j])
    elif M==4: modulation=np.array([-1-1j, -1+1j, 1+1j, 1-1j]/np.sqrt(2))
    elif M==16: modulation=np.array([-3+3j, -3+1j, -3-3j, -3-1j, 
                                 -1+3j, -1+1j, -1-3j, -1-1j,
                                  3+3j,  3+1j,  3-3j,  3-1j,
                                  1+3j,  1+1j,  1-3j,  1-1j]/np.sqrt(10))
    return modulation

def decoder_symbol(In_node, Out_node, Hiden_node, Np): #out node for ant & sym; Hiden node for ant & sym;
    outputs=[]
    loss = []; matriks=[]
    input_antenna = keras.Input(shape=(In_node,))
    x=[]
    for active in range(Np):
        x.append({})
        x[active-1][0]=Dense(Hiden_node[0], kernel_regularizer=keras.regularizers.l2(l=0.001), activation='relu')(input_antenna)
        x[active-1][0] = BatchNormalization()(x[active-1][0])
        for i in range(1,Hiden_node.shape[0]): 
            x[active-1][int(i)]=Dense(Hiden_node[i], kernel_regularizer=keras.regularizers.l2(l=0.001), activation='relu')(x[active-1][i-1])
            x[active-1][int(i)]=BatchNormalization()(x[active-1][int(i)])
        outputs.append(Dense(Out_node, kernel_regularizer=keras.regularizers.l2(l=0.001), activation='softmax', name='Symbol_'+str(active+1))(x[active-1][Hiden_node.shape[0]-1]))
        loss.append('sparse_categorical_crossentropy'); matriks.append('accuracy')
    model = keras.Model(inputs=input_antenna, outputs=outputs, name='model')
    model.compile(SGD(lr=0.005,nesterov=True), loss=loss, metrics=matriks)
    # model.summary()
    return model

def herm(matrix):
    return np.transpose(np.conjugate(matrix))
 
def H(Nr, Nt):
    return (np.random.randn(Nr,Nt)+np.random.randn(Nr,Nt)*1j)/np.sqrt(2)
    
def noise(SNR, Nr, Es):
    return (np.random.randn(Nr,1)+np.random.randn(Nr,1)*1j)*np.sqrt(Es/np.power(10,(SNR)/10))/np.sqrt(2)
