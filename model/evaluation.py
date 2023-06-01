import os
#import jsondim
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
from sklearn import metrics
from sklearn.calibration import calibration_curve
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

if not os.path.exists("./data/output"):
    os.makedirs("./data/output")
    
class Loss(nn.Module):
    def __init__(self,device,acc,ppv,sensi,tnr,npv,auroc,aurocPlot,auprc,auprcPlot,callb,callbPlot):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.classify_loss2 = nn.BCEWithLogitsLoss()
        self.device=device
        self.acc=acc
        self.ppv=ppv
        self.sensi=sensi
        self.tnr=tnr
        self.npv=npv
        self.auroc=auroc
        self.aurocPlot=aurocPlot
        self.auprc=auprc
        self.auprcPlot=auprcPlot
        self.callb=callb
        self.callbPlot=callbPlot

    def forward(self, prob, labels,logits, train=True, standalone=False):
        classify_loss='NA' 
        auc,apr='NA'
        base='NA'
        accur='NA'
        prec='NA'
        recall='NA'
        spec='NA'
        npv_val='NA'
        ECE='NA'
        MCE='NA'
        
        #prob = prob.data.cpu().numpy()
        #print(torch.sum(torch.isnan(prob)))
#         print(prob.shape)
#         print(labels.shape)
        #print(prob)
        if standalone:
            prob=torch.tensor(prob)
            labels=torch.tensor(labels)
            logits=torch.tensor(logits)
        #print(prob)
        prob=prob.type(torch.FloatTensor)
        labels=labels.type(torch.FloatTensor)
        logits=logits.type(torch.FloatTensor)
        
        pos_ind = labels >= 0.5
        neg_ind = labels < 0.5
        pos_label = labels[pos_ind]
        neg_label = labels[neg_ind]
        pos_prob = prob[pos_ind]
        neg_prob = prob[neg_ind]
        pos_loss, neg_loss = 0, 0

        #################           BCE            #######################
        if len(pos_prob):
            pos_prob=pos_prob.to(self.device)
            pos_label=pos_label.to(self.device)
            pos_loss = self.classify_loss(pos_prob, pos_label) 
       
        if len(neg_prob):
            neg_prob=neg_prob.to(self.device)
            neg_label=neg_label.to(self.device)
            neg_loss = self.classify_loss(neg_prob, neg_label)
        
        
        classify_loss = pos_loss + neg_loss
        logits=logits.to(self.device)
        labels=labels.to(self.device)
        classify_loss2 = self.classify_loss2(logits, labels)
        
        if train:
            return classify_loss
        #################           AUROC            #######################
        
        labels = labels.data.cpu().numpy()
        prob = prob.data.cpu().numpy()
        if(self.auroc):
#             print(labels)
#             print(prob)
            fpr, tpr, threshholds = metrics.roc_curve(labels, prob)
            auc = metrics.auc(fpr, tpr)
        if(self.aurocPlot):
            self.auroc_plot(labels, prob)
        
        #################           AUPRC            #######################
        if(self.auprc):
            base = ((labels==1).sum())/labels.shape[0]

            precision, recall, thresholds = metrics.precision_recall_curve(labels, prob)
            apr = metrics.auc(recall, precision)
        
        
        # stati number
        prob1 = prob >= 0.5
        #print(prob)
        
        pos_l = (labels==1).sum()
        neg_l = (labels==0).sum()
        pos_p = (prob1 + labels == 2).sum()#how many positives are predicted positive#####TP
        neg_p = (prob1 + labels == 0).sum()#True negatives
        prob2 = prob < 0.5
        fn    = (prob2 + labels==2).sum()
        fp    = (prob2 + labels==0).sum()
        #print(classify_loss, pos_p, pos_l, neg_p, neg_l)
        
        #################           Accuracy            #######################
        if(self.acc):
            accur=metrics.accuracy_score(labels,prob>=0.5)
        
        #################           Precision/PPV  (TP/(TP+FP))         #######################
        if(self.ppv):
            prec=metrics.precision_score(labels,prob>=0.5)
        
        #################           Recall/TPR/Sensitivity(TP/(TP+FN))          #######################
        if(self.sensi):
            recall=pos_p/(pos_p+fn)
        #################           Specificity/TNR  (TN/(TN+FP))         #######################
        if(self.tnr):
            spec=neg_p/(neg_p+fp)
        
        #################           NPV  (TN/(TN+FN))         #######################
        if(self.npv):
            npv_val=neg_p/(neg_p+fn)
        #################           Callibration         #######################
        if(self.callb):
            if(self.callbPlot):
                ECE, MCE = self.calb_metrics(prob,labels,True)
            else:
                ECE, MCE = self.calb_metrics(prob,labels,False)
        
        #################           Fairness         #######################
        
        
        print("BCE Loss: {:.2f}".format(classify_loss))
        print("AU-ROC: {:.2f}".format(auc))
        print("AU-PRC: {:.2f}".format(apr))
        print("AU-PRC Baaseline: {:.2f}".format(base))
        print("Accuracy: {:.2f}".format(accur))
        print("Precision: {:.2f}".format(prec))
        print("Recall: {:.2f}".format(recall))
        print("Specificity: {:.2f}".format(spec))
        print("NPV: {:.2f}".format(npv_val))
        print("ECE: {:.2f}".format(ECE))
        print("MCE: {:.2f}".format(MCE))
        
        #return [classify_loss, auc,apr,base,accur,prec,recall,spec,npv_val,ECE,MCE]
    

    def auroc_plot(self,label, pred):
        plt.figure(figsize=(8,6))
        plt.plot([0, 1], [0, 1],'r--')

        
        fpr, tpr, thresh = metrics.roc_curve(label, pred)
        auc = metrics.roc_auc_score(label, pred)
        plt.plot(fpr, tpr, label=f'Deep Learning Model, auc = {str(round(auc,3))}')


        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.title("AUC-ROC")
        plt.legend()
        plt.savefig('./data/output/'+"auroc_plot.png")
        #plt.show()
        
    def calb_curve(self,bins,bin_accs,ECE, MCE):
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')

        # Error bars
        plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

        # Draw bars and identity line
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        plt.legend(handles=[ECE_patch, MCE_patch])
        plt.savefig('./data/output/'+"callibration_plot.png")
        #plt.show()
        
    def calb_bins(self,preds,labels):
        # Assign each prediction to a bin
        num_bins = 10
        bins = np.linspace(0.1, 1, num_bins)
        binned = np.digitize(preds, bins)

        # Save the accuracy, confidence and size of each bin
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)

        for bin in range(num_bins):
            bin_sizes[bin] = len(preds[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

        return bins, binned, bin_accs, bin_confs, bin_sizes


    def calb_metrics(self,preds,labels,curve):
        ECE = 0
        MCE = 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calb_bins(preds,labels)
        
        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)
        if curve:
            self.calb_curve(bins,bin_accs,ECE, MCE)
        return ECE, MCE


