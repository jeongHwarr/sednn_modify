"""
Summary:  Calculate PESQ and overal stats of enhanced speech. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import argparse
import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import makedirs


def plot_training_stat(DIRECTORY, args, bgn_iter, fin_iter, interval_iter):
    """Plot training and testing loss. 
    
    Args: 
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files. 
    """
    workspace = DIRECTORY['WORKSPACE']
    tr_snr = args.tr_snr 
    tr_losses, te_losses, iters = [], [], []
    
    # Load stats. 
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    for iter in range(bgn_iter, fin_iter+1, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = pickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])
        
    # Plot
    line_tr, = plt.plot(tr_losses, c='b', label="Train")
    line_te, = plt.plot(te_losses, c='r', label="Test")
    plt.axis([0, len(iters), 0, max(tr_losses)])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(handles=[line_tr, line_te])
    plt.xticks(np.arange(len(iters)), iters)
    plt.show()


def calculate_pesq(DIRECTORY, args):
    """Calculate PESQ of all enhaced speech. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of clean speech. 
      te_snr: float, testing SNR. 
    """
    workspace = DIRECTORY['WORKSPACE']
    speech_dir = DIRECTORY['TE_SPEECH_DIR']
    te_snr = args.te_snr 
    
    # Remove already existed file. 
    os.system('del pesq_results.txt')
    
    # Calculate PESQ of all enhaced speech. 
    enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr))
    names = os.listdir(enh_speech_dir)
    for (cnt, na) in enumerate(names):
        print(cnt, na)
        enh_path = os.path.join(enh_speech_dir, na)
        
        speech_na = na.split('.')[0]
        speech_path = os.path.join(speech_dir, "%s.WAV" % speech_na)
        
        # Call executable PESQ tool. 
        cmd = ' '.join(["pesq2.exe", speech_path, enh_path, '+'+str(args.sample_rate)])
#        os.system(cmd)  
        result = os.popen(cmd).read()
        print(result)
        
def get_stats(DIRECTORY, args):
    """Calculate stats of PESQ. 
    """
    workspace = DIRECTORY['WORKSPACE']
    pesq_path = "pesq_results.txt"
    with open(pesq_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    pesq_dict = {}
    for i1 in range(1, len(lis) - 1):
        li = lis[i1]
        na = li[1]
        pesq = float(li[2])
        noise_type = na.split('.')[1]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
        else:
            pesq_dict[noise_type].append(pesq)
        
    avg_list, std_list = [], []
    result_path = os.path.join(workspace, "result")
    makedirs(result_path)
    result_path = os.path.join(result_path,"result.txt")
    file = open(result_path, "w")
    f = "{0:<16} {1:<16}"
    file.write(f.format("Noise", "PESQ")+"\n")
    file.write("---------------------------------\n")
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        file.write(f.format(noise_type, "%.2f +- %.2f\n" % (avg_pesq, std_pesq)))
    file.write("---------------------------------\n")
    file.write(f.format("Avg.", "%.2f +- %.2f\n" % (np.mean(avg_list), np.mean(std_list))))
    file.close()
    print("Average PESQ score: %s" %np.mean(avg_list))
