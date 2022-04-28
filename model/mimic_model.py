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
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,med_signal,lab_signal,embed_size,rnn_size,batch_size):
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
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        if self.med_vocab_size:
            self.med=CodeBase(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeBase(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeBase(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeBase(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeBase(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1
       
        
        #self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        
       # self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+2*self.rnn_size, 1, False)
        self.fc1=nn.Linear(int(self.rnn_size*self.modalities), int((self.rnn_size*self.modalities)/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*self.modalities)/2), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,procs,outs,charts,labs,conds,demo,contrib):        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
#         conds=conds.to(self.device)
#         conds=self.condEmbed(conds)
#         print(conds.shape)
#         conds=conds.view(conds.shape[0],-1)
#         print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
       
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
    
    
    
class LSTMBaseH(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,med_signal,lab_signal,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
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
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        if self.med_vocab_size:
            self.med=CodeBase(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeBase(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeBase(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeBase(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeBase(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1

        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        self.condfc=nn.Linear((self.embed_size*self.cond_seq_len),self.rnn_size, False)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        #self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        self.fc1=nn.Linear(int(self.rnn_size*(self.modalities+2)), int((self.rnn_size*(self.modalities+2))/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*(self.modalities+2))/2), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,procs,outs,charts,labs,conds,demo,contrib):         
        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        conds=self.condfc(conds)
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
        
        out1=torch.cat((out1,conds),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
        
            

class CodeBase(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,signal,lab):             
        super(CodeBase, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.signal=signal
        self.lab_sig=lab
        self.build()
    
    def build(self):
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        if self.signal: 
            self.codeRnn = nn.LSTM(input_size=int(self.embed_size*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 1,batch_first=True)
        else:
            self.codeRnn = nn.LSTM(input_size=int((self.embed_size+1)*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 1,batch_first=True)
        
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

        if code.shape[0]==2:
            dat=code[1]
            code=code[0]
            if self.lab_sig:
                if code.shape[1]>500:
                    code=code[:,0:500,:]
                    dat=dat[:,0:500,:]
            #print(code.shape)
            #print(dat.shape)
            codeEmbedded=self.codeEmbed(code)
            #code=self.codeEmbed(code)
            #code=torch.transpose(code,1,2)
            #code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
            #print(code.shape)
            #print(self.signal)
            if not self.signal:
                if self.lab_sig:
                    #codeEmbedded=0
                    test=torch.max(code,2)
                    test=test.values
                    test=test.unsqueeze(2)
                    code=torch.zeros(code.shape[0],code.shape[1],code.shape[2])
                    code=code.type(torch.FloatTensor)
                    code=code.to(self.device)
                    test=test.type(torch.FloatTensor)
                    test=test.to(self.device)
                    code=torch.add(code,test)
                    code=code.type(torch.LongTensor)
                    code=code.to(self.device)
                    codeEmbedded=self.codeEmbed(code)
                #code=0
                dat=dat.unsqueeze(3)
                #print(dat.shape)
                dat=dat.type(torch.FloatTensor)
                dat=dat.to(self.device)
                #print(dat.shape)
                #print(codeEmbedded.shape)
                codeEmbedded=torch.cat((codeEmbedded,dat),3)
            code=torch.transpose(codeEmbedded,1,2)
            #code=torch.transpose(code,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            
                #print(code.shape)
        else:
            code=self.codeEmbed(code)
            code=torch.transpose(code,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
        #code=torch.transpose(code,1,2)
        #print(code[0])
        #print(dat[0])
        #print(code[0,:,:])
        
        
        
        #print(code[0,0:2,0:3,:])
        
        #code=torch.sum(code,1)
        
                
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
        
        code_h_n=code_h_n.squeeze()
        #print("output",code_h_n.shape)
        
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
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,med_signal,lab_signal,embed_size,rnn_size,batch_size):
        super(LSTMAttn, self).__init__()
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
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        
        if self.med_vocab_size:
            self.med=CodeAttn(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeAttn(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeAttn(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeAttn(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeAttn(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1

        
        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        self.condfc=nn.Linear((self.embed_size*self.cond_seq_len),self.rnn_size, False)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        #self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        self.fc1=nn.Linear(int(self.rnn_size*(self.modalities+2)), int((self.rnn_size*(self.modalities+2))/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*(self.modalities+2))/2), int((self.rnn_size*(self.modalities+2))/4), False)
        self.fc3=nn.Linear(int((self.rnn_size*(self.modalities+2))/4), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,procs,outs,charts,labs,conds,demo,contrib):        
        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        conds=self.condfc(conds)
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
        
        out1=torch.cat((out1,conds),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
        
            


# In[ ]:


class CodeAttn(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,signal,lab):           
        super(CodeAttn, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.signal=signal
        self.build()
        self.lab_sig=lab
    
    def build(self):
        
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        if self.signal: 
            self.codeRnn = nn.LSTM(input_size=int(self.embed_size*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
            #self.codeRnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
        else:
            self.codeRnn = nn.LSTM(input_size=int((self.embed_size+1)*self.code_seq_len),hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)
            #self.codeRnn = nn.LSTM(input_size=self.embed_size+1,hidden_size=self.rnn_size,num_layers = 2,dropout=0.2,batch_first=True)

        self.code_fc=nn.Linear(self.rnn_size, 1, False)
        #self.dropout1 = nn.Dropout(0.2)
        
    def forward(self, code):
        #print(conds.shape)

        h_0, c_0 = self.init_hidden()
        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

        #Embedd all sequences
        #print(code.shape)
        #print(code[0,:,:])

        if code.shape[0]==2:
            dat=code[1]
            code=code[0]
            if self.lab_sig:
                if code.shape[1]>500:
                    code=code[:,0:500,:]
                    dat=dat[:,0:500,:]
            codeEmbedded=self.codeEmbed(code)
            #code=torch.transpose(code,1,2)
            #code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
            #print(code.shape)
            #print(self.signal)
            if not self.signal:
                if self.lab_sig:
                    test=torch.max(code,2)
                    test=test.values
                    test=test.unsqueeze(2)
                    code=torch.zeros(code.shape[0],code.shape[1],code.shape[2])
                    code=code.type(torch.FloatTensor)
                    code=code.to(self.device)
                    test=test.type(torch.FloatTensor)
                    test=test.to(self.device)
                    code=torch.add(code,test)
                    code=code.type(torch.LongTensor)
                    code=code.to(self.device)
                    codeEmbedded=self.codeEmbed(code)
                dat=dat.unsqueeze(3)
                #print(dat.shape)
                dat=dat.type(torch.FloatTensor)
                dat=dat.to(self.device)
                codeEmbedded=torch.cat((codeEmbedded,dat),3)
            code=torch.transpose(codeEmbedded,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(codeEmbedded,1)
            
                #print(code.shape)
        else:
            code=self.codeEmbed(code)
            code=torch.transpose(code,1,2)
            code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
            #code=torch.sum(code,1)
        #print(code.shape)
        #code=torch.transpose(code,1,2)
        #print(code[0])
        #print(dat[0])
        #print(code[0,:,:])

        h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)
        #print(code.shape)
        #code=code.type(torch.FloatTensor)
#        code_time=code_time.type(torch.FloatTensor)
        #h_0, c_0, code = h_0.to(self.device), c_0.to(self.device),code.to(self.device)

#        code=torch.cat((code,code_time),dim=2)
            
        #Run through LSTM
        code_output, (code_h_n, code_c_n)=self.codeRnn(code, (h_0, c_0))
        #print("code_output",code_output.shape)
        
        code_softmax=self.code_fc(code_output)
        #code_output=self.dropout1(code_output) 
        #print("softmax",code_softmax.shape)
        code_softmax=F.softmax(code_softmax)
        #print("softmax",code_softmax.shape)
        code_softmax=torch.sum(torch.mul(code_output,code_softmax),dim=1)
        #print("softmax",code_softmax.shape)
        #print("========================")
        
        return code_softmax
    
    
    def init_hidden(self):
        # initialize the hidden state and the cell state to zeros
        h=torch.zeros(2,self.batch_size, self.rnn_size)
        c=torch.zeros(2,self.batch_size, self.rnn_size)

#         if self.hparams.on_gpu:
#             hidden_a = hidden_a.cuda()
#             hidden_b = hidden_b.cuda()

        h = Variable(h)
        c = Variable(c)

        return (h, c)    
    

            
class CNNBase(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,med_signal,lab_signal,embed_size,rnn_size,batch_size):
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
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        if self.med_vocab_size:
            self.med=CodeCNN(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeCNN(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeCNN(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeCNN(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeCNN(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1
            
        self.fc1=nn.Linear(int(self.rnn_size*self.modalities), int((self.rnn_size*self.modalities)/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*self.modalities)/2), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,procs,outs,charts,labs,conds,demo,contrib):        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
#         conds=conds.to(self.device)
#         conds=self.condEmbed(conds)
#         print(conds.shape)
#         conds=conds.view(conds.shape[0],-1)
#         print(conds.shape)
        #print("cond_pool_ob",cond_pool_ob.shape)
        #out1=torch.cat((cond_pool,cond_pool_ob),1)
        #out1=cond_pool
       
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
    
    
    
class CNNBaseH(nn.Module):
    def __init__(self,device,cond_vocab_size,cond_seq_len,proc_vocab_size,proc_seq_len,med_vocab_size,med_seq_len,out_vocab_size,out_seq_len,chart_vocab_size,chart_seq_len,lab_vocab_size,lab_seq_len,eth_vocab_size,gender_vocab_size,age_vocab_size,med_signal,lab_signal,embed_size,rnn_size,batch_size): #proc_vocab_size,med_vocab_size,lab_vocab_size
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
        self.lab_vocab_size=lab_vocab_size
        self.lab_seq_len=lab_seq_len
        if self.chart_seq_len>500:
            self.chart_seq_len=500
        self.batch_size=batch_size
        self.padding_idx = 0
        self.modalities=0
        self.device=device
        self.med_signal,self.lab_signal=med_signal,lab_signal
        self.build()
        
    def build(self):
        if self.med_vocab_size:
            self.med=CodeCNN(self.device,self.embed_size,self.rnn_size,self.med_vocab_size,self.med_seq_len,self.batch_size,self.med_signal,False)
            self.modalities=self.modalities+1
                
        if self.proc_vocab_size:
            self.proc=CodeCNN(self.device,self.embed_size,self.rnn_size,self.proc_vocab_size,self.proc_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.out_vocab_size:
            self.out=CodeCNN(self.device,self.embed_size,self.rnn_size,self.out_vocab_size,self.out_seq_len,self.batch_size,True,False)
            self.modalities=self.modalities+1
        if self.chart_vocab_size:
            self.chart=CodeCNN(self.device,self.embed_size,self.rnn_size,self.chart_vocab_size,self.chart_seq_len,self.batch_size,self.lab_signal,True)
            self.modalities=self.modalities+1
        if self.lab_vocab_size:
            self.lab=CodeCNN(self.device,self.embed_size,self.rnn_size,self.lab_vocab_size,self.lab_seq_len,self.batch_size,self.lab_signal,False)
            self.modalities=self.modalities+1

        self.condEmbed=nn.Embedding(self.cond_vocab_size,self.embed_size,self.padding_idx) 
        self.condfc=nn.Linear((self.embed_size*self.cond_seq_len),self.rnn_size, False)
        
        self.ethEmbed=nn.Embedding(self.eth_vocab_size,self.embed_size,self.padding_idx) 
        self.genderEmbed=nn.Embedding(self.gender_vocab_size,self.embed_size,self.padding_idx) 
        self.ageEmbed=nn.Embedding(self.age_vocab_size,self.embed_size,self.padding_idx) 
        self.demo_fc=nn.Linear(self.embed_size*3, self.rnn_size, False)
        
        #self.fc=nn.Linear((self.embed_size*self.cond_seq_len)+3*self.rnn_size, 1, False)
        self.fc1=nn.Linear(int(self.rnn_size*(self.modalities+2)), int((self.rnn_size*(self.modalities+2))/2), False)
        self.fc2=nn.Linear(int((self.rnn_size*(self.modalities+2))/2), int((self.rnn_size*(self.modalities+2))/4), False)
        self.fc3=nn.Linear(int((self.rnn_size*(self.modalities+2))/4), 1, False)
        
        #self.sig = nn.Sigmoid()
        
    def forward(self,meds,procs,outs,charts,labs,conds,demo,contrib):         
        
        out1 = torch.zeros(size=(1,0))
        
        if len(meds[0]):
            med_h_n = self.med(meds)  
            med_h_n=med_h_n.view(med_h_n.shape[0],-1)
            #print("med_h_n",med_h_n.shape)
            out1=med_h_n
            #print(out1.shape)
            #print(out1.nelement())
        if len(procs):
            proc_h_n = self.proc(procs)  
            proc_h_n=proc_h_n.view(proc_h_n.shape[0],-1)
            #print("proc_h_n",proc_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,proc_h_n),1)
            else:
                out1=proc_h_n
        if len(labs[0]):
            lab_h_n = self.lab(labs)  
            lab_h_n=lab_h_n.view(lab_h_n.shape[0],-1)
            #print("lab_h_n",lab_h_n.shape)
            if out1.nelement():
                out1=torch.cat((out1,lab_h_n),1)
            else:
                out1=lab_h_n
        if len(outs):
            out_h_n = self.out(outs)  
            out_h_n=out_h_n.view(out_h_n.shape[0],-1)
            if out1.nelement():
                out1=torch.cat((out1,out_h_n),1)
            else:
                out1=out_h_n
        if len(charts[0]):
            chart_h_n = self.chart(charts)  
            chart_h_n=out_h_n.view(chart_h_n.shape[0],-1)
            if out1.nelement:
                out1=torch.cat((out1,chart_h_n),1)
            else:
                out1=chart_h_n
        
        conds=conds.to(self.device)
        conds=self.condEmbed(conds)
        #print(conds.shape)
        conds=conds.view(conds.shape[0],-1)
        conds=self.condfc(conds)
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
        
        out1=torch.cat((out1,conds),1)
        out1=torch.cat((out1,demog),1)
        #print("out1",out1.shape)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)
        #print("out1",out1.shape)
        
        sig = nn.Sigmoid()
        sigout1=sig(out1)
        #print("sig out",sigout1[16])
        #print("sig out",sigout1)
        #print(out1[0])
        #print("hi")
        
        return sigout1,out1
        
        
            


# In[ ]:


class CodeCNN(nn.Module):
    def __init__(self,device,embed_size,rnn_size,code_vocab_size,code_seq_len,batch_size,signal,lab):             
        super(CodeCNN, self).__init__()
        self.embed_size=embed_size
        self.rnn_size=rnn_size
        self.code_vocab_size=code_vocab_size
        self.code_seq_len=code_seq_len
        self.batch_size=batch_size
        self.padding_idx = 0
        self.device=device
        self.signal=signal
        self.lab_sig=lab
        self.build()
    
    
    def build(self):
        self.codeEmbed=nn.Embedding(self.code_vocab_size,self.embed_size,self.padding_idx)
        if self.signal: 
            self.conv1 = nn.Conv1d(self.embed_size,self.rnn_size, kernel_size = 10, stride = 1, padding =0)
        else:
            self.conv1 = nn.Conv1d(int(self.embed_size+1),self.rnn_size, kernel_size = 10, stride = 1, padding =0)
            
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
        
        if code.shape[0]==2:
            dat=code[1]
            code=code[0]
            if self.lab_sig:
                if code.shape[1]>500:
                    dat=dat[:,0:500,:]
                    code=code[:,0:500,:]
            code=self.codeEmbed(code)
            #print(code.shape)
            if not self.signal:
                dat=dat.unsqueeze(3)
                #print(dat.shape)
                dat=dat.type(torch.FloatTensor)
                dat=dat.to(self.device)
                code=torch.cat((code,dat),3)
            code=torch.sum(code,1)
            code=code.permute(0,2,1)
            #code=torch.transpose(code,1,2)
            #code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
                #print(code.shape)
        else:
            code=self.codeEmbed(code)
            code=torch.sum(code,1)
            code=code.permute(0,2,1)
            #code=torch.transpose(code,1,2)
            #code=torch.reshape(code,(code.shape[0],code.shape[1],-1))
        #code=torch.transpose(code,1,2)
        #print(code[0])
        #print(dat[0])
        #print(code[0,:,:])
        
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
    

