import numpy as np
import pickle
import os.path as osp
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
# import sys
# import os
# sys.path.append(os.getcwd())
from defenses import datasets
from tqdm import tqdm

from pulp import *

# from defenses.victim import *

class incremental_kmeans():
    def __init__(self,blackbox,epsilon,ydist='l1',optim="approx",trainingset_name=None,frozen = True):
        self.blackbox = blackbox
        self.device = self.blackbox.device
        self.out_path = self.blackbox.out_path
        self.num_classes = self.blackbox.num_classes
        self.log_path = self.blackbox.log_path
        self.log_prefix = self.blackbox.log_prefix
        
        self.epsilon = epsilon
        if ydist == 'l1':
            self.norm = 1
        elif ydist == 'l2':
            self.norm = 2
        else:
            raise RuntimeError("Not supported distance metrics for y distance!")
        self.optim=optim
        self.frozen = bool(frozen)
        self.labels = []
        self.centroids = []

        if self.out_path is not None:
            centroids_path = osp.join(self.out_path,'quantization_centroids.pickle')
            if osp.exists(centroids_path): # load labels directly
                print("Loading centroids from "+centroids_path) 
                with open(centroids_path,'rb') as f:
                    self.centroids = pickle.load(f)
            
            label_path = osp.join(self.out_path,'quantization_label.pickle')
            if osp.exists(label_path):
                print("Loading labels from "+label_path)
                with open(label_path,'rb') as f:
                    self.labels = pickle.load(f)
        
        if len(self.labels) == 0 and trainingset_name is not None: # generate labels from blackbox
            modelfamily = datasets.dataset_to_modelfamily[trainingset_name]
            transform_type = 'test'
            transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
            trainingset = datasets.__dict__[trainingset_name](train=True, transform=transform)
            dataloader = DataLoader(trainingset,batch_size=32,num_workers=4,shuffle=False)
            print("training quantizer with training set "+trainingset_name)
            self.labels = [[] for _ in range(self.num_classes)]
            for data,label in tqdm(dataloader):
                y_prime = self.blackbox(data,stat=False)
                for n,l in enumerate(label.numpy()):
                    self.labels[l].append(y_prime[n,:].reshape([1,-1]))
            for l in range(self.num_classes):
                self.labels[l] = torch.cat(self.labels[l],dim=0)
            with open(label_path,'wb') as f:
                pickle.dump([label.cpu() for label in self.labels],f)
        
        if len(self.centroids) == 0 and len(self.labels) > 0:
            for l in range(len(self.labels)):
                self.centroids.append(self.cluster(self.labels[l]))
                

            with open(centroids_path,'wb') as f:
                pickle.dump([centroid.cpu() for centroid in self.centroids],f)
            
        for l in range(len(self.centroids)):
            print("Initialized quantizer: label %d has %d centroids!"%(l,len(self.centroids[l])))

        
        for n in range(len(self.labels)):
            self.labels[n] = self.labels[n].to(self.blackbox.device)
            self.centroids[n] = self.centroids[n].to(self.blackbox.device)

            
        
        if len(self.centroids)==0:    
            print("[Warning]: Using empty centroids as initialization!")

        self.queries = []
        self.quantize_queries = []
        self.call_count = 0
        if self.log_path is not None:
            self.quantize_log_path = self.log_path.replace('distance','quantize_distance')
            if not osp.exists(self.quantize_log_path):
                with open(self.quantize_log_path, 'w') as wf:
                    columns = ['call_count', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                    wf.write('\t'.join(columns) + '\n')
        else:
            self.quantize_log_path = None
    
    def set_centroids(self,centroids):
        self.centroids = deepcopy(centroids)

    def get_cluster_centroid(self,cluster_data):
        if self.norm == 2:
            # For l2 norm, we can get the centroid with the mean directly
            centroid = torch.mean(cluster_data,dim=0,keepdim=True)
        
        elif self.norm == 1: 
            # For l1 norm, we have two methods: approx and exact
            data = cluster_data.detach().cpu().numpy()
            max_idx = np.argmax(data[0,:])
            n,K = data.shape
            if self.optim == 'approx':
                # solve the centroids with least l1 norm sum approximately
                # if no simplex constraints, the median will be the optimal solution
                median = np.median(data,axis=0) 
                
                # project the median back to the simplex constraints with minimal l1 distance
                prob = LpProblem("simplex projection",LpMinimize)

                # build variable
                c = LpVariable.dicts("centroid",list(range(K)),lowBound=0.0,upBound=1.0)
                z = LpVariable.dicts("absolutevalue",list(range(K)),lowBound=0.0)
                
                # add objective function
                # minimize l1 perturbation
                prob += lpSum([z[i] for i in range(K)])

                # add constraints
                for i in range(K):
                    # slack variable constraint: z >= |c-median|
                    prob += z[i]-c[i]>=-median[i], "z[{}]>=c[{}]-median[{}]".format(i,i,i)
                    prob += z[i]+c[i]>=median[i], "z[{}]>=median[{}]-c[{}]".format(i,i,i)
                    
                    # argmax constraint
                    if i != max_idx:
                        prob += c[max_idx]-c[i]>=1e-4, "c[{}]>=c[{}]".format(max_idx,i)
                
                # simplex constraint
                prob += lpSum([c[i] for i in range(K)])==1.0, "Simplex Constraint"

                prob.solve()
                centroid = np.zeros([1,K])
                for i in range(K):
                    centroid[0,i]=c[i].varValue
                
                centroid = torch.tensor(centroid).to(cluster_data)
                
            elif self.optim == 'exact':
                # solve the controid with exactly the lowest l1 distance sum
                print("[Warning]: Solving for exact l1 centroid, which may take a long time!")

                prob = LpProblem("L1 Centroid",LpMinimize)

                # build variable
                c = LpVariable.dicts("centroid",list(range(K)),lowBound=0.0,upBound=1.0)
                z = LpVariable.dicts("absolutevalue",list(range(n*K)),lowBound=0.0)

                # add objective function
                # minimize total l1 distance
                prob += lpSum([z[i] for i in range(n*K)])

                # add constraints
                for i in range(n):
                    for j in range(K):
                        # slack variable constraints: z[i] >= |data[i]-c|
                        prob += z[i*K+j]+c[j]>=data[i,j], "z[{0},{1}]>=data[{0},{1}]-c[{1}]".format(i,j)
                        prob += z[i*K+j]-c[j]>=-data[i,j], "z[{0},{1}]>=c[{1}]-data[{0},{1}]".format(i,j)

                # argmax constraint
                for j in range(K):
                    if j != max_idx:
                        prob += c[max_idx]-c[j]>=1e-4, "c[{}]>=c[{}]".format(max_idx,j)

                # simplex constraint
                prob += lpSum([c[j] for j in range(K)])==1.0, "Simplex Constraint"

                prob.solve()
                centroid = np.zeros([1,K])
                for j in range(K):
                    centroid[0,j]=c[j].varValue
                
                centroid = torch.tensor(centroid).to(data)
            
            else:
                raise RuntimeError("Not recognized optimization method: "+self.optim)

        else:
            raise RuntimeError("Not supported ydist")
        return centroid

    def quantize(self,input,centroids):
        # print(input.shape)
        if len(input.shape)==1:
            input = input.reshape([1,-1]) # make it as N x D
        
        
        distance_matrix = []
        for i in range(len(centroids)):
            distance_matrix.append(torch.norm(input-centroids[i,:].reshape([1,-1]),p=self.norm,dim=1,keepdim=True))
        distance_matrix = torch.cat(distance_matrix,dim=1)
        distance,cent_idxs = torch.min(distance_matrix,dim=1)
        return distance,cent_idxs

    
    def k_means(self,data,inital_centroids,tolerance=1e-3,max_iter=100):
        centroids = deepcopy(inital_centroids)
        for _ in range(max_iter):
            new_centroids = deepcopy(centroids)
            _,cent_idxs = self.quantize(data,centroids) # cluster
            for c in range(len(centroids)):
                new_centroids[c] = self.get_cluster_centroid(data[cent_idxs==c]) # mean/median
            max_move = torch.max(torch.norm(new_centroids-centroids,p=2,dim=1))
            centroids = new_centroids
            if max_move<tolerance:
                break
        
        return centroids

    def cluster(self,data):
        """
        perform cluster with data and return centroids
        """
        if len(data)==0:
            return []
        
        print("Reclustering!")
        
        centroids = self.get_cluster_centroid(data)
        
        while True:
            val_distance, _ = self.quantize(data,centroids)
            # cents = centroids[cent_idxs]
            # val_distance = torch.norm(data-cents,p=self.norm,dim=1)
            # mean_dis = torch.mean(val_distance,dim=0)
            max_dis,max_idx = torch.max(val_distance,dim=0)

            if max_dis>self.epsilon: # keep adding new centroids until the constraint is satisfied
                centroids = torch.cat([centroids,data[max_idx,:].reshape([1,-1])],dim=0)
                centroids = self.k_means(data,centroids)
            else:
                break
        return centroids
    


    def __call__(self,input,train=False,stat=True,return_origin=False):
        
        y_prime,y_v = self.blackbox(input,stat=False,return_origin=True) # go through the blackbox first

        max_idx = torch.argmax(y_prime,dim=1)
        cents = torch.zeros_like(y_prime)
        if train and not self.frozen:
            if len(self.labels)<self.num_classes:
                self.labels = self.labels + [[] for _ in range(self.num_classes-len(self.labels))]
            if len(self.centroids)<self.num_classes:
                self.centroids = self.centroids + [[] for _ in range(self.num_classes-len(self.centroids))]
            
            for n in range(self.num_classes):
                if torch.sum(max_idx==n)>0:
                    # put the new y_prime into the dataset by their top-1 labels
                    if len(self.labels[n]) == 0:
                        self.labels[n] = y_prime[max_idx==n,:]
                    else:
                        self.labels[n] = torch.cat([self.labels[n],y_prime[max_idx==n,:]],dim=0)
                    # initializa the centroids for this label if there is no centroids for this label currently
                    if len(self.centroids[n])==0 and len(self.labels[n])>0:
                        self.centroids[n] = self.cluster(self.labels[n])

                    
                    # Get Quantizer and check if utility constraints are satisfied
                    _,cent_idxs = self.quantize(y_prime[max_idx==n,:],self.centroids[n])
                    cents[max_idx==n,:] = self.centroids[n][cent_idxs]
                    val_distance = torch.norm(y_prime[max_idx==n,:]-cents[max_idx==n,:],p=self.norm,dim=1)
                    max_dis = torch.max(val_distance)
                    if max_dis>self.epsilon: # re-cluster with new data if the utility constraints are not satisfied
                        self.centroids[n] = self.cluster(self.labels[n])
                        _,cent_idxs = self.quantize(y_prime[max_idx==n,:],self.centroids[n])
                        cents[max_idx==n,:] = self.centroids[n][cent_idxs]
        
        else: # freeze the centroids and only quantize
            for n in range(self.num_classes):
                if torch.sum(max_idx==n)>0:
                    # Get Quantizer and check if utility constraints are satisfied
                    _,cent_idxs = self.quantize(y_prime[max_idx==n,:],self.centroids[n])
                    cents[max_idx==n,:] = self.centroids[n][cent_idxs]

        # Sanity checks
       
        # Constraints are met
        if not self.blackbox.is_in_simplex(cents):
            print('[WARNING] Simplex contraint failed (i = {})'.format(self.call_count))
        
        if stat and self.log_path is not None:
            self.call_count += len(y_v)
            self.queries.append((y_v.cpu().detach().numpy(), cents.cpu().detach().numpy()))
            self.quantize_queries.append((y_prime.cpu().detach().numpy(), cents.cpu().detach().numpy()))
            if self.call_count % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                quantize_query_out_path = osp.join(self.out_path, 'quantize_queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)
                
                with open(quantize_query_out_path,'wb') as wf:
                    pickle.dump(self.quantize_queries, wf)

                l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.blackbox.calc_query_distances(self.queries)
                
                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

                
                l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.blackbox.calc_query_distances(self.quantize_queries)
                with open(self.quantize_log_path, 'a') as af:
                    test_cols = [self.call_count, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')


        if return_origin:
            return cents,y_v
        else:
            return cents


    def get_yprime(self,y):
        y = self.blackbox.get_yprime(y)
        max_idx = torch.argmax(y,dim=1)

        cents = torch.zeros_like(y)
        for n in range(self.num_classes):
            if torch.sum(max_idx==n)>0:
                # Get Quantizer and check if utility constraints are satisfied
                _,cent_idxs = self.quantize(y[max_idx==n,:],self.centroids[n])
                cents[max_idx==n,:] = self.centroids[n][cent_idxs]
        return cents

# if __name__ == '__main__':
    
#     bb = Blackbox(torch.nn.Linear(10,10))
#     quantizer = incremental_kmeans(bb,1.0)

#     data = torch.rand([5,3])
#     data = data/torch.sum(data,dim=1,keepdim=True)
#     data,_ = torch.sort(data,dim=1)
#     print(data)
#     cent = quantizer.cluster(data)

#     dis,idx = quantizer.quantize(data,cent)

#     print(cent)
#     print(dis)
#     print(idx)
