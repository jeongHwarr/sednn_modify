import os
import argparse
from work_module import prepare_data, main_dnn, evaluate 

CURRENT_PATH = os.path.split(os.getcwd())[-1]

MINIDATA = True #MINIDATA

if MINIDATA == True:
    WORKSPACE = os.path.join("D:/Python_output",CURRENT_PATH+"_MINIDATA")
    TR_SPEECH_DIR="mini_data/train_speech"
    TR_NOISE_DIR="mini_data/train_noise"
    TE_SPEECH_DIR="mini_data/test_speech"
    TE_NOISE_DIR="mini_data/test_noise"
    
else:    
    WORKSPACE = "D:/Python_output/"+CURRENT_PATH
    TR_SPEECH_DIR="D:/train/speech"
    TR_NOISE_DIR="D:/noise"
    TE_SPEECH_DIR="D:/test"
    TE_NOISE_DIR="D:/noise"   

def get_args():
    parser = argparse.ArgumentParser(description="Speech Enhancement using DNN.")
    parser.add_argument('-sr', '--sample_rate', default=8000, type=int,
                     help="target sampling rate of audio")
    parser.add_argument('--fft', default=256, type=int,
                         help="FFT size")
    parser.add_argument('--window', default=256, type=int,
                         help="window size")
    parser.add_argument('--overlap', default=192, type=int,
                     help="overlap size of spectrogram") 
    parser.add_argument('--n_concat', default=7, type=int,
                     help="number of frames to concatentate")
   
    parser.add_argument('--tr_snr', default=0, type=int, 
                        help="SNR of training data")
    parser.add_argument('--te_snr', default=0, type=int, 
                        help="SNR of test data") 
    
    parser.add_argument('--iter', default=10000, type=int,
                        help="number of iteration for training")
    parser.add_argument('--debug_inter', default=1000, type=int, 
                        help="Interval to debug model")
    parser.add_argument('--save_inter', default=5000, type=int, 
                        help="Interval to save model")
    parser.add_argument('-b', '--batch_size', default=32, type=int)  
    parser.add_argument('--lr', default=0.0001, type=float,
                         help="Initial learning rate")
    parser.add_argument('-visual', '--visualize', default=1, type=int, choices=[0,1],
                        help="If value is 1, visualization of result of inference") 

    parser.add_argument('--train', default=1, type=int, choices=[0,1],
                        help="If the value is 1, run training") 
    parser.add_argument('--test', default=1, type=int, choices=[0,1],
                         help="If the value is 1, run test")

    args = parser.parse_args() 
    return args

if __name__ == '__main__':

    args = get_args()

    DIRECTORY = {}   
    DIRECTORY['WORKSPACE'] = WORKSPACE
    DIRECTORY['TR_SPEECH_DIR'] = TR_SPEECH_DIR
    DIRECTORY['TR_NOISE_DIR'] = TR_NOISE_DIR
    DIRECTORY['TE_SPEECH_DIR'] = TE_SPEECH_DIR
    DIRECTORY['TE_NOISE_DIR'] = TE_NOISE_DIR

    assert args.iter >= args.debug_inter and args.iter >=args.save_inter, "Number of training iterations should greater than or equal to the debugging and store interval"
     
    prepare_data.create_mixture_csv(DIRECTORY, args, mode='train')
    prepare_data.create_mixture_csv(DIRECTORY, args, mode='test')
    
    prepare_data.calculate_mixture_features(DIRECTORY, args, mode='train')
    prepare_data.calculate_mixture_features(DIRECTORY, args, mode='test')
    
    prepare_data.pack_features(DIRECTORY, args, mode='train')
    prepare_data.pack_features(DIRECTORY, args, mode='test')
    
    if args.train==1:
        prepare_data.compute_scaler(DIRECTORY, args, mode='train')
        main_dnn.train(DIRECTORY, args)
        evaluate.plot_training_stat(DIRECTORY, args, bgn_iter=0, fin_iter=args.iter, interval_iter=args.debug_inter)
        
    if args.test==1:
        main_dnn.inference(DIRECTORY, args)
        evaluate.calculate_pesq(DIRECTORY,args)
        evaluate.get_stats(DIRECTORY,args)
        
