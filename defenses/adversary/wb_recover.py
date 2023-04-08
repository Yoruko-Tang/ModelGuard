from turtle import distance
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm
import csv
import os.path as osp
import pickle
import numpy as np

from torch.multiprocessing import Process,Manager

from defenses.utils.model import soft_cross_entropy
numclasses_to_nn = {
    10:[64,64],
    100:[512,512],
    200:[1024,1024],
    256:[1024,1024]
}

numclasses_to_alpha = {
    10:10,
    100:100,
    200:1000,
    256:1000
}

class Recover_NN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        hidden_layer=numclasses_to_nn[num_classes]
        self.fc1 = nn.Linear(num_classes,hidden_layer[0])
        self.fc2 = nn.Linear(hidden_layer[0],hidden_layer[1])
        self.fc3 = nn.Linear(hidden_layer[1],num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self,x):
        self.eval()
        output = self.forward(x)
        output = F.softmax(output,dim=1)
        return output



class Table_Recover():
    max_sample_size=5000000
    def __init__(self,blackbox,table_size=1000000,batch_size=1,epsilon=None,perturb_norm=1,recover_mean=True,recover_norm=2,tolerance=1e-4,recover_nn=False,alpha=None,recover_proc=1):
        self.table_size = table_size
        self.blackbox = blackbox
        self.num_classes = self.blackbox.num_classes
        self.device = self.blackbox.device
        self.batch_size = batch_size
        self.epsilon=epsilon
        self.perturb_norm = perturb_norm
        self.recover_mean=recover_mean
        self.recover_norm = recover_norm
        self.tolerance = tolerance
        self.recover_nn = recover_nn
        self.alpha = alpha
        self.num_proc = recover_proc
        self.true_label_sample,self.perturbed_label_sample = None, None
        
        

    def generate_lookup_table(self,load_path=None,estimation_set=None):
        table_size = self.table_size
        if load_path is not None:
            if osp.exists(load_path):
                with open(load_path,'rb') as wf:
                    self.true_label_sample,self.perturbed_label_sample = pickle.load(wf)
                    print("Loaded Existing Table with table length {}!".format(len(self.true_label_sample)))
                    if len(self.true_label_sample)>=table_size:
                        self.true_label_sample,self.perturbed_label_sample = self.true_label_sample[:table_size,:],self.perturbed_label_sample[:table_size,:]
                        table_size = 0
                    else:
                        table_size = table_size-len(self.true_label_sample)
                        print("Supplementing existing table with {} samples...".format(table_size))
                               
        if table_size>0:
            print("Building Recover Table! Total Samples Number={}!".format(table_size))
            
            true_label_sample = self.get_dirichlet_samples(self.alpha,table_size,estimation_set)


            if self.num_proc == 1:
                perturbed_label_sample = self.get_perturbed_label_sample(self.blackbox,true_label_sample,self.batch_size)
            else:
                perturbed_label_sample = self.get_perturbed_label_sample_parallel(self.blackbox,true_label_sample,self.num_proc)
            
            if self.epsilon is not None:
                pert_norm = torch.norm(true_label_sample-perturbed_label_sample,p=self.perturb_norm,dim=1)
                true_label_sample = true_label_sample[pert_norm<=self.epsilon]
                perturbed_label_sample = perturbed_label_sample[pert_norm<=self.epsilon]
            if self.true_label_sample is None or self.perturbed_label_sample is None:
                self.true_label_sample,self.perturbed_label_sample = true_label_sample,perturbed_label_sample
            else:
                self.true_label_sample = torch.cat([self.true_label_sample,true_label_sample],dim=0)
                self.perturbed_label_sample = torch.cat([self.perturbed_label_sample,perturbed_label_sample],dim=0)
 
            print("Recover Table Completed!")
        
            with open(osp.join(self.blackbox.out_path, 'recover_table.pickle'), 'wb') as wf:
                pickle.dump([self.true_label_sample,self.perturbed_label_sample], wf)
        try:
            self.true_label_sample,self.perturbed_label_sample = self.true_label_sample.to(self.device),self.perturbed_label_sample.to(self.device)
        except:
            print("[Warning]: Not enough GPU memory for storing the lookup table, will use cpu instead!")
            # self.device = torch.device('cpu')
            self.true_label_sample,self.perturbed_label_sample = self.true_label_sample.cpu(),self.perturbed_label_sample.cpu()

        if not self.recover_nn:
            self.true_top1 = torch.argmax(self.true_label_sample,dim=1).to(self.device)
            self.log_path = osp.join(self.blackbox.out_path, 'recover_distance{}.log.tsv'.format(self.blackbox.log_prefix))
            self.logger = csv.writer(open(self.log_path,'w'),delimiter='\t')
            self.logger.writerow(['call count','recover distance mean', 'recover distance std'])
            self.call_count = 0
        else:
            print("Training NN for Recovering!")
            self.nn = Recover_NN(self.num_classes)
            print(self.nn)
            self.nn.to(self.device)
            self.log_path = osp.join(self.blackbox.out_path, 'recover_nn_training.log.tsv')
            self.logger = csv.writer(open(self.log_path,'w'),delimiter='\t')
            self.logger.writerow(['Epoch','Loss', 'L2 Distance'])
            self.nn = self.train_recover_nn(self.nn,self.perturbed_label_sample,self.true_label_sample,epoch=100,batch_size=50000,lr=1e-3)

    def estimate_dir(self,estimation_set):
        if isinstance(estimation_set,str) and osp.exists(estimation_set): # use labels in a transfer set
            print("Estimating Dirichlet Distribution via Lables in '{}'".format(estimation_set))
            with open(estimation_set,'rb') as wf:
                estimation_data = pickle.load(wf)
                estimation_label = torch.cat([torch.tensor(estimation_data[i][1]).reshape([1,-1]) for i in range(len(estimation_data))],dim=0)
        elif isinstance(estimation_set,torch.Tensor): # use labels in a tensor
            estimation_label = estimation_set.clone().detach()
        else:
            raise RuntimeError("Not a valid estimation set form (must be in a path or a tensor)")
        return estimation_label.to(self.device)

    def get_dirichlet_samples(self,alpha=None,table_size=1000000,estimation_set=None):
        """
        Generate samples from dirichlet distribution
        """
        sample_list = []
        if estimation_set is not None:
            concentration = self.num_classes*4
            alpha = self.estimate_dir(estimation_set)*concentration
        elif alpha is None:# preset alphas
            alpha = [k*torch.ones(self.num_classes).to(self.device)/self.num_classes for k in [1.0,]]
        s = table_size//len(alpha)

        for a in alpha:
            distribution = Dirichlet(a)
            group_num = s//self.max_sample_size # the maximum size of samples is 5000000 in one generation
            final_group = s%self.max_sample_size
            for _ in range(group_num):
                samples = distribution.sample((self.max_sample_size,)).cpu()
                sample_list.append(samples)
            sample_list.append(distribution.sample((final_group,)).cpu())
            
            # samples = distribution.sample((s,)).cpu()
            # sample_list.append(samples)
        return torch.cat(sample_list,dim=0)

    def get_uniform_samples(self,table_size=1000000):
        """
        Generate samples uniformly
        """
        raise NotImplementedError("Not implemented uniform sampling")


    @staticmethod
    def get_perturbed_label_sample(blackbox,true_label_sample,batch_size=32,output=None,count=None,proc_idx=None):
        perturbed_label_sample = []
        if count is None:
            pbar = tqdm(total=len(true_label_sample))
        #with tqdm(total=len(true_label_sample)) as pbar:
        for start_idx in range(0,len(true_label_sample),batch_size):
            end_idx = min([start_idx+batch_size,len(true_label_sample)])
            perturbed_label = blackbox.get_yprime(true_label_sample[start_idx:end_idx,:])
            perturbed_label_sample.append(perturbed_label)
            if count is not None:
                count.value += len(perturbed_label)
            else:
                pbar.update(len(perturbed_label))
                

        if output is not None and proc_idx is not None:
            output[proc_idx] = torch.cat(perturbed_label_sample,dim=0)
            return
        else:
            return torch.cat(perturbed_label_sample,dim=0)


    def get_perturbed_label_sample_parallel(self,blackbox,true_label_sample,num_proc=10):
        print("Generating recover table with %d processes..."%num_proc)
        if hasattr(blackbox,'cpu'):
            blackbox.cpu()
        with Manager() as manager:
            proc_data = np.array_split(true_label_sample,num_proc)
            count = manager.Value('i',0)
            perturbed_label_output = manager.list([None,]*num_proc)
            # for i in range(num_proc):
            #     perturbed_label_output.append(None)
            proc = []
            for i in range(num_proc):
                p = Process(target=Table_Recover.get_perturbed_label_sample,args=(blackbox,proc_data[i],self.batch_size,perturbed_label_output,count,i))
                proc.append(p)
            for p in proc:
                p.start()
            with tqdm(total=len(true_label_sample)) as pbar:
                prev_count = 0
                while None in perturbed_label_output:
                    current_count = count.value
                    if current_count>prev_count:
                        pbar.update(current_count-prev_count)
                        prev_count=current_count
                
                current_count = sum([len(perturbed_label_output[n]) for n in range(num_proc)])
                if current_count>prev_count:
                    pbar.update(current_count-prev_count)
                    

            
            for p in proc:
                p.join()
                # p.terminate()
            
            perturbed_label_sample = list(perturbed_label_output)
        res = torch.cat(perturbed_label_sample,dim=0)
        print("Ended multiprocessing with total number of samples = %d"%len(res))
        if hasattr(blackbox,'to_blackbox_device'):
            blackbox.to_blackbox_device()
        return res

    def train_recover_nn(self,model,pert_label,true_label,epoch=30,batch_size=50000,lr=1e-2):
        dataset = TensorDataset(pert_label,true_label)
        trainloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        optimizer = optim.Adam(model.parameters(),lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
        model.train()
        for e in tqdm(range(epoch)):
            for pl,tl in trainloader:
                pl,tl = pl.to(self.device),tl.to(self.device)
                output = model(pl)
                loss = soft_cross_entropy(output,tl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            l2_dist = torch.mean(torch.norm(model.predict(pl)-tl,p=2,dim=1)).cpu().item()
            print("Epoch: {}\tLoss: {:.4f}\tL2 Distance: {:.4f}".format(e+1,loss.cpu().item(),l2_dist))
            self.logger.writerow([e+1,loss.cpu().item(),l2_dist])

        
        return model

    def __call__(self, yprime,pbar=None):
        assert yprime.dim()==2, "yprime must be a batch with dim=2"
        yprime = yprime.to(self.device) # all operations in this model is performed on CPU to avoid memory running out
        
        if self.recover_nn:
            y = self.nn(yprime)
        else:
            res = torch.zeros_like(yprime).to(self.device)
            rec_dis = []
            top1_label = torch.argmax(yprime,dim=1)
            for c in range(self.num_classes):
                y = []
                yprime_c = yprime[top1_label==c]
                perturbed_label_filtered = self.perturbed_label_sample[self.true_top1==c,:]
                true_label_filtered = self.true_label_sample[self.true_top1==c,:]
                perturbed_label_filtered,true_label_filtered = perturbed_label_filtered.to(self.device),true_label_filtered.to(self.device)
                for i in range(len(yprime_c)):
                    # return the true lable with the same top-1 label and minimal perturbed distance
                    distances = torch.norm(yprime_c[i]-perturbed_label_filtered,p=self.recover_norm,dim=1)
                    if not self.recover_mean:
                        min_idx = torch.argmin(distances)
                        y.append(true_label_filtered[min_idx,:].unsqueeze(0))
                        rec_dis.append(distances[min_idx])
                    else:
                        tolerance = max([self.tolerance,torch.min(distances).cpu().item()])
                        y.append(torch.mean(true_label_filtered[distances<=tolerance,:],dim=0,keepdim=True))
                        rec_dis.append(torch.mean(distances[distances<=tolerance]))
                    if pbar is not None:
                        pbar.update(1)
                
                res[top1_label==c]=torch.cat(y,dim=0)
                    
            self.call_count += len(yprime)
            mean_rec_dis = torch.mean(torch.tensor(rec_dis)).cpu().item()
            std_rec_dis = torch.std(torch.tensor(rec_dis)).cpu().item()
            self.logger.writerow([self.call_count,mean_rec_dis,std_rec_dis])
            
        
        return res
            

