from turtle import distance
import torch
import torch.optim as optim
from torch.distributions import Dirichlet
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm
import csv
import os.path as osp
import pickle
import numpy as np

from torch.multiprocessing import Process,Manager

class Table_Recover():
    max_sample_size=5000000
    def __init__(self,blackbox,table_size=1000000,batch_size=1,recover_mean=False,tolerance=1e-4,dir=True,alpha=None,estimation_set=None,num_proc=1):
        self.table_size = table_size
        self.blackbox = blackbox
        self.device = self.blackbox.device
        self.batch_size = batch_size
        self.recover_mean=recover_mean
        self.tolerance = tolerance

        
        table_path = osp.join(self.blackbox.out_path, 'recover_table.pickle')
        if osp.exists(table_path):
            print("Loading Existing Table!")
            with open(table_path,'rb') as wf:
                self.true_label_sample,self.perturbed_label_sample = pickle.load(wf)
                
        else:
            print("Building Recover Table! Total Samples Number={}!".format(table_size))
            if dir:
                self.true_label_sample = self.get_dirichlet_samples(alpha,table_size,estimation_set)
            else:
                self.true_label_sample = self.get_uniform_samples(table_size)

            if num_proc == 1:
                self.perturbed_label_sample = self.get_perturbed_label_sample(self.blackbox,self.true_label_sample,self.batch_size)
            else:
                self.perturbed_label_sample = self.get_perturbed_label_sample_parallel(self.blackbox,self.true_label_sample,num_proc)
 
            print("Recover Table Completed!")
        
            with open(table_path, 'wb') as wf:
                pickle.dump([self.true_label_sample,self.perturbed_label_sample], wf)
        self.true_label_sample,self.perturbed_label_sample = self.true_label_sample.cpu(),self.perturbed_label_sample.cpu()
        self.true_top1 = torch.argmax(self.true_label_sample,dim=1)
        self.log_path = osp.join(self.blackbox.out_path, 'recover_distance{}.log.tsv'.format(self.blackbox.log_prefix))
        self.logger = csv.writer(open(self.log_path,'w'),delimiter='\t')
        self.logger.writerow(['call count','recover distance mean', 'recover distance std'])

        self.call_count = 0

    def estimate_dir(self,estimation_set,lr=1e-3,batch_size=None,max_epoch=100,epsilon=1e-4):
        print("Estimating Dirichlet Distribution via Lables in '{}'".format(estimation_set))
        with open(estimation_set,'rb') as wf:
            estimation_data = pickle.load(wf)
            estimation_label = torch.cat([torch.tensor(estimation_data[i][1]).reshape([1,-1]) for i in range(len(estimation_data))],dim=0)
            estimation_dataloader = DataLoader(TensorDataset(estimation_label),batch_size=batch_size if batch_size is not None else len(estimation_data),shuffle=True)
        alpha = torch.ones(estimation_label.size(1)).to(self.device)
        alpha.requires_grad_()
        Dir = Dirichlet(alpha)
        optimizer = optim.Adam([alpha,],lr = lr)
        for e in range(max_epoch):
            for _,label in estimation_dataloader:
                old_alpha = alpha.clone().detach()
                label = label.to(self.device)
                optimizer.zero_grad()
                log_prob = Dir.log_prob(label)
                loss = -torch.mean(log_prob)

                loss.backward()
                optimizer.step()

                dist = torch.norm(alpha.data-old_alpha,p=2).item()
                if dist<epsilon:
                    break
            
            print("Epoch:{}\tNeg Log Prob Loss:{:.4f}".format(e,loss.item()))
            if dist<epsilon:
                break
        return alpha.detach_()

    def get_dirichlet_samples(self,alpha=None,table_size=1000000,estimation_set=None):
        """
        Generate samples from dirichlet distribution
        """
        sample_list = []
        if estimation_set is not None and osp.exists(estimation_set):
            alpha = [self.estimate_dir(estimation_set),]
        elif alpha is None:# preset alphas
            alpha = [k*torch.ones(self.blackbox.num_classes).to(self.device)/self.blackbox.num_classes for k in [1.0,]]
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
    def get_perturbed_label_sample(blackbox,true_label_sample,batch_size=32,output=None,count=None):
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
                

        if output is not None:
            output.append(torch.cat(perturbed_label_sample,dim=0))
            return
        else:
            return torch.cat(perturbed_label_sample,dim=0)


    def get_perturbed_label_sample_parallel(self,blackbox,true_label_sample,num_proc=10):
        print("Generating recover table with %d processes..."%num_proc)
        with Manager() as manager:
            proc_data = np.array_split(true_label_sample,num_proc)
            count = manager.Value('i',0)
            perturbed_label_output = manager.list()
            proc = []
            for i in range(num_proc):
                p = Process(target=Table_Recover.get_perturbed_label_sample,args=(blackbox,proc_data[i],self.batch_size,perturbed_label_output,count))
                proc.append(p)
            for p in proc:
                p.start()
            with tqdm(total=len(true_label_sample)) as pbar:
                prev_count = 0
                while len(perturbed_label_output) < num_proc:
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
        return res



    def __call__(self, yprime):
        assert yprime.dim()==2, "yprime must be a batch with dim=2"
        yprime = yprime.cpu() # all operations in this model is performed on CPU to avoid memory running out
        y = []
        rec_dis = []
        top1_label = torch.argmax(yprime,dim=1)
        for i in range(len(yprime)):
            # return the true lable with the same top-1 label and minimal perturbed distance
            perturbed_label_filtered = self.perturbed_label_sample[self.true_top1==top1_label[i],:]
            true_label_filtered = self.true_label_sample[self.true_top1==top1_label[i],:]
            distances = torch.sum((yprime[i]-perturbed_label_filtered)**2,dim=1)
            if not self.recover_mean:
                min_idx = torch.argmin(distances)
                y.append(true_label_filtered[min_idx,:].unsqueeze(0))
                rec_dis.append(distances[min_idx])
            else:
                tolerance = max([self.tolerance,torch.min(distances).cpu().item()])
                y.append(torch.mean(true_label_filtered[distances<=tolerance,:],dim=0,keepdim=True))
                rec_dis.append(torch.mean(distances[distances<=tolerance]))
                
        self.call_count += len(yprime)
        mean_rec_dis = torch.mean(torch.tensor(rec_dis)).cpu().item()
        std_rec_dis = torch.std(torch.tensor(rec_dis)).cpu().item()
        self.logger.writerow([self.call_count,mean_rec_dis,std_rec_dis])
        y = torch.cat(y,dim=0).to(self.device)
        
        return y
            

