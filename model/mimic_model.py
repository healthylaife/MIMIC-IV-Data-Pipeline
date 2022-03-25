import os
#import jsondim
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
from sklearn import metrics
import numpy as np
from collections import defaultdict
import sys


class LSTMBase(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,embed_size,rnn_size,batch_size):
        super(LSTMBase, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=1
        self.device=device
        self.build()
        
    def build(self):
        self.med=CodeBase(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size)
        self.proc=CodeBase(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size)
       
        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        
        self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+2*self.rnn_size, 1, False)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, meds,procs,conds,contrib):        
        
        med_h_n = self.med(meds)  
        med_h_n=med_h_n.view(med_h_n.shape[0],-1)
        print("med_h_n",med_h_n.shape)
        
        proc_h_n = self.proc(procs)  
        proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
        print("proc_h_n",proc_h_n.shape)
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        out1=torch.cat((conds,med_h_n),1)
        out1=torch.cat((out1,proc_h_n),1)
        print("out1",out1.shape)
        out1 = self.fc(out1)
        #print("out1",out1.shape)
        
        sigout1 = self.sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1
    
    
    
class LSTMBaseH(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
        super(LSTMBaseH, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=1
        self.device=device
        self.build()
        
    def build(self):
        self.med=CodeBase(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size)
        self.proc=CodeBase(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size)

        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, meds,procs,conds,demo,contrib):        
        
        med_h_n = self.med(meds)  
        med_h_n=med_h_n.view(med_h_n.shape[0],-1)
        print("med_h_n",med_h_n.shape)
        
        proc_h_n = self.proc(procs)  
        proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
        print("proc_h_n",proc_h_n.shape)
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        eth=demo[0].to(self.device)
        eth=self.ethEmbed(eth)
        
        gender=demo[1].to(self.device)
        gender=self.genderEmbed(gender)
        
        age=demo[2].to(self.device)
        age=self.ageEmbed(age)
        
        demog=torch.cat((eth,gender),1)
        demog=torch.cat((demog,age),1)
        #print("demog",demog.shape)
        demog=self.demo_fc(demog)
        
        out1=torch.cat((conds,med_h_n),1)
        out1=torch.cat((out1,proc_h_n),1)
        out1=torch.cat((out1,demog),1)
        print("out1",out1.shape)
        out1 = self.fc(out1)
        #print("out1",out1.shape)
        
        sigout1 = self.sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1
        
            


# In[ ]:


class CodeBase(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,bmi_flag=False):             
        super(CodeBase, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
    
    def build(self):

        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        self.codeRnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.rnn_size,num_layers = 1,batch_first=True)
        
    def forward(self, code):
        #print(conds.shape)
        #ob=code[2]
#        code_time=code[1]
#        code=code[0]
        #print()
        #initialize hidden and cell state
        h_0, c_0 = self.init_hidden()
        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #Embedd all sequences
        print(code.shape)
        #print(code[0,:,:])
        
        #code=torch.transpose(code,1,2)
        #print(code.shape)
        #print(code[0,:,:])
        
        code=self.codeEmbed(code)
        #print(code.shape)
        #print(code[0,0:2,0:3,:])
        
        code=torch.sum(code,1)
        #print(code.shape)
        #code=code.view(code.shape[0],code.shape[1],-1)
        print(code.shape)
        #print(code[0,0:2,0:15])
        #print(code[0,:,:])

        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #code=code.type(torch.FloatTensor)
#        code_time=code_time.type(torch.FloatTensor)
        #h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

#        code=torch.cat((code,code_time),dim=2)
            
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.codeRnn(code, (h_0, c_0))
        
        code_h_n=code_h_n.squeeze()
        print("output",code_h_n.shape)
        
        return code_h_n
    
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(1,self.batch_size, self.rnn_size)
        c=torch.zeros(1,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    

class LSTMAttn(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
        super(LSTMAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=1
        self.device=device
        self.build()
        
    def build(self):

        self.med=CodeAttn(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size)
        self.proc=CodeAttn(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size)

        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        self.cond_fc=nn.Linear(self.rnn_size, 1, False)
        
        self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+2*self.rnn_size, 1, False)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, meds,procs,conds,contrib):        
        
        med_h_n = self.med(meds[0])  
        med_h_n=med_h_n.view(med_h_n.shape[0],-1)
        #print("med_h_n",med_h_n.shape)
        
        proc_h_n = self.proc(procs)  
        proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
        print("proc_h_n",proc_h_n.shape)
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        #print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        out1=torch.cat((conds,med_h_n),1)
        out1=torch.cat((out1,proc_h_n),1)
        #print("out1",out1.shape)
        out1 = self.fc(out1)
        #print("out1",out1.shape)
        
        sigout1 = self.sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1
        
            


# In[ ]:


class CodeAttn(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size):             
        super(CodeAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
    
    def build(self):

        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        self.codeRnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.rnn_size,num_layers = 1,batch_first=True)
        self.code_fc=nn.Linear(self.rnn_size, 1, False)
        
    def forward(self, code):
        #print(conds.shape)

        #initialize hidden and cell state
        h_0, c_0 = self.init_hidden()
        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #Embedd all sequences
        #print(code.shape)
        #print(code[0,:,:])
        
        #code=torch.transpose(code,1,2)
        #print(code.shape)
        #print(code[0,:,:])
        
        code=self.codeEmbed(code)
        #print(code.shape)
        #print(code[0,0:2,0:3,:])
        
        code=torch.sum(code,1)
        #print(code.shape)
        #code=code.view(code.shape[0],code.shape[1],-1)
        #print(code.shape)
        #print(code[0,0:2,0:15])
        #print(code[0,:,:])

        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #code=code.type(torch.FloatTensor)
#        code_time=code_time.type(torch.FloatTensor)
        #h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

#        code=torch.cat((code,code_time),dim=2)
            
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.codeRnn(code, (h_0, c_0))
        #print("code_output",code_output.shape)
        
        code_softmax=self.code_fc(code_output)
        #print("softmax",code_softmax.shape)
        code_softmax=F.softmax(code_softmax)
        #print("softmax",code_softmax.shape)
        code_softmax=torch.sum(torch.mul(code_output,code_softmax),dim=1)
        #print("softmax",code_softmax.shape)
        #print("========================")
        
        return code_softmax
    
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(1,self.batch_size, self.rnn_size)
        c=torch.zeros(1,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    
    
    
class LSTMAttnH(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
        super(LSTMAttnH, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=1
        self.device=device
        self.build()
        
    def build(self):

        self.med=CodeAttn(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size)
        self.proc=CodeAttn(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size)

        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        #self.cond_fc=nn.Linear(self.embed_size, 1, False)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, meds,procs,conds,demo,contrib):        
        
        med_h_n = self.med(meds[0])  
        med_h_n=med_h_n.view(med_h_n.shape[0],-1)
        #print("med_h_n",med_h_n.shape)
        proc_h_n = self.proc(procs)  
        proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
        print("proc_h_n",proc_h_n.shape)
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        #print(conds.shape)
        #print("demo[0]",demo[0].shape)
        
        eth=demo[0].to(self.device)
        eth=self.ethEmbed(eth)
        
        gender=demo[1].to(self.device)
        gender=self.genderEmbed(gender)
        
        age=demo[2].to(self.device)
        age=self.ageEmbed(age)
        
        demog=torch.cat((eth,gender),1)
        demog=torch.cat((demog,age),1)
        #print("demog",demog.shape)
        demog=self.demo_fc(demog)
        #print("demog",demog.shape)
        
        out1=torch.cat((conds,med_h_n),1)
        out1=torch.cat((out1,proc_h_n),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc(out1)
        #print("out1",out1.shape)
        
        sigout1 = self.sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1
        
            
class CNNBase(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
        super(CNNBase, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=1
        self.device=device
        self.build()
        
    def build(self):
        self.med=CodeCNN(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size)
        self.proc=CodeCNN(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size)
       
        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        
        self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+2*self.rnn_size, 1, False)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, meds,procs,conds,contrib):        
        
        med_h_n = self.med(meds[0])  
        med_h_n=med_h_n.view(med_h_n.shape[0],-1)
        print("med_h_n",med_h_n.shape)
        
        proc_h_n = self.proc(procs)  
        proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
        print("proc_h_n",proc_h_n.shape)
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        out1=torch.cat((conds,med_h_n),1)
        out1=torch.cat((out1,proc_h_n),1)
        print("out1",out1.shape)
        out1 = self.fc(out1)
        #print("out1",out1.shape)
        
        sigout1 = self.sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1
    
    
    
class CNNBaseH(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
        super(CNNBaseH, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.eth_vocab_size=eth_vocab_size
        self.gender_vocab_size=gender_vocab_size
        self.age_vocab_size=age_vocab_size
        self.cond_vocab_size=cond_vocab_size
        self.cond_seq_len=cond_seq_len
        self.proc_vocab_size=proc_vocab_size
        self.proc_seq_len=proc_seq_len
        self.med_vocab_size=med_vocab_size
        self.med_seq_len=med_seq_len
        self.out_vocab_size=out_vocab_size
        self.out_seq_len=out_seq_len
        self.chart_vocab_size=chart_vocab_size
        self.chart_seq_len=chart_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=1
        self.device=device
        self.build()
        
    def build(self):
        self.med=CodeCNN(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size)
        self.proc=CodeCNN(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size)

        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, meds,procs,conds,demo,contrib):        
        
        med_h_n = self.med(meds[0])  
        med_h_n=med_h_n.view(med_h_n.shape[0],-1)
        #print("med_h_n",med_h_n.shape)
        
        proc_h_n = self.proc(procs)  
        proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
        #print("proc_h_n",proc_h_n.shape)
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        #print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
        eth=demo[0].to(self.device)
        eth=self.ethEmbed(eth)
        
        gender=demo[1].to(self.device)
        gender=self.genderEmbed(gender)
        
        age=demo[2].to(self.device)
        age=self.ageEmbed(age)
        
        demog=torch.cat((eth,gender),1)
        demog=torch.cat((demog,age),1)
        #print("demog",demog.shape)
        demog=self.demo_fc(demog)
        
        out1=torch.cat((conds,med_h_n),1)
        out1=torch.cat((out1,proc_h_n),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc(out1)
        #print("out1",out1.shape)
        
        sigout1 = self.sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1
        
            


# In[ ]:


class CodeCNN(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,bmi_flag=False):             
        super(CodeCNN, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.build()
    
    def build(self):

        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        self.conv1 = nn.Conv1d(self.embed_size,self.rnn_size, kernel_size = 3, stride = 1, padding =0)
        self.bn1 = nn.BatchNorm1d(self.rnn_size)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1, True)
        
    def forward(self, code):
        #print(conds.shape)
        #ob=code[2]
#        code_time=code[1]
#        code=code[0]
        #print()
        #initialize hidden and cell state
        h_0, c_0 = self.init_hidden()
        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #Embedd all sequences
        #print(code.shape)
        #print(code[0,:,:])
        
        #code=torch.transpose(code,1,2)
        #print(code.shape)
        #print(code[0,:,:])
        
        code=self.codeEmbed(code)
        #print(code.shape)
        #print(code[0,0:2,0:3,:])
        
        code=torch.sum(code,1)
        code=code.permute(0,2,1)
        #print(code.shape)
        #code=code.view(code.shape[0],code.shape[1],-1)
        #print(code.shape)
        #print(code[0,0:2,0:15])
        #print(code[0,:,:])

        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #code=code.type(torch.FloatTensor)
#        code_time=code_time.type(torch.FloatTensor)
        #h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

#        code=torch.cat((code,code_time),dim=2)
        
        #Run through LSTM
        code_output = self.conv1(code)
        #print("output",cond_output.shape)
        code_output = self.bn1(code_output)
        #print("output",code_output.shape)
        
        code_pool, code_indices = self.maxpool1(code_output)
        #print("output",code_pool.shape)
        
        
        code_pool = torch.squeeze(code_pool)
        code_indices = torch.squeeze(code_indices)
        
        
        return code_pool
    
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(1,self.batch_size, self.rnn_size)
        c=torch.zeros(1,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    

