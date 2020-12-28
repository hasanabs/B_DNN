# B-DNN GSM-MIMO Detector

This is a simulation program of Generalized Spatial Modulation (GSM) detector using DNN as the base of the signal detector.
Comparing with the conventional methods, the proposed method has better performance of BER. Also, since this method based on DNN that can easily parallelize due to the availability of computation equipment, so the time complexity of the proposed method is also reliable.

This compute capsule uses:
- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [TensorFlow 2](https://www.tensorflow.org/)
- [Matplotlib](https://matplotlib.org/)

## Quick Start Guide
There are 4 Detectors in this simulation. For all detectors, you just need change parameters inside "Change specification inside this area" to simulate custom specification. 

If you want to simulate B-DNN detector you can run B-DNN_Prediction.py, if a message like this "The model with specification Np=xx Nr=yy M=zz is not available yet, you might run B-DNN_Training.py with that specification before running B-DNN_Prediction.py file", you have to complete training step first. 

Some trained parameter are available for this template that is:
- Np=2, Nr=2, M=2,4,16
- Np=2, Nr=4, M=2,4,16


## Citations

It would be highly appreciated if you cite the following reference for your work:
- [1] H. Albinsaid, K. Singh, S. Biswas, C. -P. Li and M. -S. Alouini, "Block Deep Neural Network-Based Signal Detector for Generalized Spatial Modulation," in IEEE Communications Letters, vol. 24, no. 12, pp. 2775-2779, Dec. 2020, doi: [10.1109/LCOMM.2020.3015810](https://ieeexplore.ieee.org/document/9165095).

