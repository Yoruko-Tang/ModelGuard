from turtle import distance
import torch
from torch.distributions import Dirichlet
from tqdm import tqdm
import csv
import os.path as osp
import pickle
class Table_Recover():
    def __init__(self,blackbox,table_size=1000000,batch_size=1,recover_mean=False,tolerance=1e-4):
        self.table_size = table_size
        alpha = [0.001,0.01,0.1,0.3,1.0,3.0,10.0,30.0,100.0,1000.0]
        s = table_size//len(alpha)
        self.blackbox = blackbox
        self.device = self.blackbox.device
        self.batch_size = batch_size
        self.recover_mean=recover_mean
        self.tolerance = tolerance
        true_label_sample = []
        perturbed_label_sample = []
        
        table_path = osp.join(self.blackbox.out_path, 'recover_table.pickle')
        if osp.exists(table_path):
            print("Loading Existing Table!")
            with open(table_path,'rb') as wf:
                self.true_label_sample,self.perturbed_label_sample = pickle.load(wf)
        else:
            print("Building Recover Table! Total Samples Number={}!".format(table_size))
            for a in alpha:
                self.distribution = Dirichlet(a*torch.ones(self.blackbox.num_classes)/self.blackbox.num_classes)
                samples = self.distribution.sample((s,)).to(self.device)
                true_label_sample.append(samples)
            self.true_label_sample = torch.cat(true_label_sample,dim=0)
            with tqdm(total=table_size) as pbar:
                for start_idx in range(0,table_size,self.batch_size):
                    end_idx = min([start_idx+batch_size,table_size])
                    perturbed_label_sample.append(blackbox.get_yprime(self.true_label_sample[start_idx:end_idx,:]))
                    pbar.update(end_idx-start_idx)

            
            self.perturbed_label_sample = torch.cat(perturbed_label_sample,dim=0)
            print("Recover Table Completed!")
        
            with open(table_path, 'wb') as wf:
                pickle.dump([self.true_label_sample,self.perturbed_label_sample], wf)
        self.true_top1 = torch.argmax(self.true_label_sample,dim=1)
        self.log_path = osp.join(self.blackbox.out_path, 'recover_distance{}.log.tsv'.format(self.blackbox.log_prefix))
        self.logger = csv.writer(open(self.log_path,'w'),delimiter='\t')
        self.logger.writerow(['call count','recover distance mean', 'recover distance std'])

        self.call_count = 0

    def __call__(self, yprime):
        assert yprime.dim()==2, "yprime must be a batch with dim=2"
        yprime = yprime.to(self.device)
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
        y = torch.cat(y,dim=0)
        
        return y
            

