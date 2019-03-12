# Speech enhancement using deep neural networks (Keras implementation)
### by Yong Xu and Qiuqiang Kong
### Modified Jeonghwa Yoo (Env: python 3.5 and windows OS)

This code uses deep neural network (DNN) to do speech enhancement. This code is a Keras implementation of The paper:

[1] Xu, Y., Du, J., Dai, L.R. and Lee, C.H., 2015. A regression approach to speech enhancement based on deep neural networks. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 23(1), pp.7-19.

Original C++ implementation is here (https://github.com/yongxuUSTC/DNN-for-speech-enhancement) by Yong Xu (yong.xu@surrey.ac.uk). 

Original Keras re-implementation(https://github.com/yongxuUSTC/sednn/tree/master/mixture2clean_dnn) is done by Qiuqiang Kong (q.kong@surrey.ac.uk)

<pre>
Noise(0dB)   PESQ
----------------------
n64     1.36 +- 0.05
n71     1.35 +- 0.18
----------------------
Avg.    1.35 +- 0.12
</pre>

## Run on TIMIT and 115 noises
You may replace the mini data with your own data. We listed the data need to be prepared in meta_data/ to re-run the experiments in [1]. The data contains:

Training:
Speech: TIMIT 4620 training sentences. 
Noise: 115 kinds of noises (http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/115noises.html)

Testing:
Speech: TIMIT 168 testing sentences (selected 10% from 1680 testing sentences)
Noise: Noise 92 (http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex.html)

Some of the dataset are not published. Instead, you could collect your own data. 

1. Download and prepare data. 

2. Set MINIDATA=0 in run.py. Modify WORKSPACE, TR_SPEECH_DIR, TR_NOISE_DIR, TE_SPEECH_DIR, TE_NOISE_DIR in run.py and some arguments (get_args() function) 

3. Run run.py

<pre>
Iteration: 0, tr_loss: 1.228049, te_loss: 1.252313
Iteration: 1000, tr_loss: 0.533825, te_loss: 0.677872
Iteration: 2000, tr_loss: 0.505751, te_loss: 0.678816
Iteration: 3000, tr_loss: 0.483631, te_loss: 0.666576
Iteration: 4000, tr_loss: 0.480287, te_loss: 0.675403
Iteration: 5000, tr_loss: 0.457020, te_loss: 0.676319
Saved model to /vol/vssp/msos/qk/workspaces/speech_enhancement/models/0db/md_5000iters.h5
Iteration: 6000, tr_loss: 0.461330, te_loss: 0.673847
Iteration: 7000, tr_loss: 0.445159, te_loss: 0.668545
Iteration: 8000, tr_loss: 0.447244, te_loss: 0.680740
Iteration: 9000, tr_loss: 0.427652, te_loss: 0.678236
Iteration: 10000, tr_loss: 0.421219, te_loss: 0.663294
Saved model to /vol/vssp/msos/qk/workspaces/speech_enhancement/models/0db/md_10000iters.h5
Training time: 202.551192045 s
</pre>

The final PESQ looks like:

<pre>
Noise(0dB)            PESQ
---------------------------------
pink             2.01 +- 0.23
buccaneer1       1.88 +- 0.25
factory2         2.21 +- 0.21
hfchannel        1.63 +- 0.24
factory1         1.93 +- 0.23
babble           1.81 +- 0.28
m109             2.13 +- 0.25
leopard          2.49 +- 0.23
volvo            2.83 +- 0.23
buccaneer2       2.03 +- 0.25
white            2.00 +- 0.21
f16              1.86 +- 0.24
destroyerops     1.99 +- 0.23
destroyerengine  1.86 +- 0.23
machinegun       2.55 +- 0.27
---------------------------------
Avg.             2.08 +- 0.24
</pre>


## Visualization
In the inference step, you may add --visualize to the arguments to plot the mixture, clean and enhanced speech log magnitude spectrogram. 

![alt text](https://github.com/yongxuUSTC/deep_learning_based_speech_enhancement_keras_python/blob/master/mixture2clean_dnn/appendix/enhanced_log_sp.png)

## PESQ (windows OS) from
https://uk.mathworks.com/matlabcentral/fileexchange/47333-pesq-matlab-driver

## Bugs report:
1. PESQ dose not support long path/folder name, so please shorten your path/folder name. Or you will get a wrong/low PESQ score (or you can modify the PESQ source code to enlarge the size of the path name variable)

2. For larger dataset which can not be loaded into the momemory at one time, you can 1. prepare your training scp list ---> 2. random your training scp list ---> 3. split your triaining scp list into several parts ---> 4. read each part for training one by one
