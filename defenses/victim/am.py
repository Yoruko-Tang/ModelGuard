import os.path as osp
import json
import numpy as np
import torch
import torch.nn.functional as F
import defenses.models.zoo as zoo
from defenses import datasets
from defenses.victim import Blackbox
import pickle

class AM(Blackbox):
    def __init__(self, model, model_def, defense_levels=0.99, device=None, num_classes=10, rand_fhat=10, use_adaptive=True, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        print('=> AM ({})'.format([self.dataset_name, defense_levels]))

        self.device = torch.device('cuda') if device is None else device
        self.model = model.to(device)
        self.model.eval()
        model_def = model_def.to(device)
        self.num_classes = num_classes
        self.defense_fn = selectiveMisinformation(model_def, defense_levels, self.num_classes, rand_fhat, use_adaptive)


    @classmethod
    def from_modeldir(cls, model_dir, device=None, rand_fhat=False, use_adaptive=True, output_type='probs', **kwargs):
        device = torch.device('cuda') if device is None else device
        param_path = osp.join(model_dir, 'params.json')
        with open(param_path) as jf:
            params = json.load(jf)
        model_arch = params["model_arch"]
        num_classes = params["num_classes"]
        if 'queryset' in params:
            dataset_name = params['queryset']
        elif 'testdataset' in params:
            dataset_name = params['testdataset']
        elif 'dataset' in params:
            dataset_name = params['dataset']
        modelfamily = datasets.dataset_to_modelfamily[dataset_name]

        model = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)
        model_def = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)

        model = model.to(device)

        # Load Weights
        checkpoint_path = osp.join(model_dir, 'model_best.pth.tar')
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, "checkpoint.pth.tar")
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        best_test_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc)
        )
        model_def_path = model_dir + '/model_poison.pt'
        if not rand_fhat:
            print("loading misinformation model")
            model_def.load_state_dict(torch.load(model_def_path))
        model_def = model_def.to(device)
        print(
            "=> loaded checkpoint for Misinformation model (used for Selective Misinformation)"
        )

        blackbox = cls(model=model, model_def=model_def,
                       output_type=output_type,
                       dataset_name=dataset_name,
                       defense_levels=0.99, device=device, num_classes=num_classes,
                       rand_fhat=rand_fhat,
                       use_adaptive=use_adaptive)
        return blackbox

    def __call__(self, x, T=1, return_origin=False, stat=False):
        with torch.no_grad():
            x = x.to(self.device)
            y = self.model(x)
            self.call_count += x.shape[0]
            y = F.softmax(y/T, dim=1)
        y_mod = self.defense_fn(x, y)
        if stat:
            self.queries.append((y.cpu().detach().numpy(), y_mod.cpu().detach().numpy()))

            if self.call_count % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)

                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.calc_query_distances(self.queries)

                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if return_origin:
            return y_mod, y 
        else:
            return y_mod


def compute_hellinger(y_a, y_b):
    """
    :param y_a: n x K dim
    :param y_b: n x K dim
    :return: n dim vector of hell dist between elements of y_a, y_b
    """
    diff = torch.sqrt(y_a) - torch.sqrt(y_b)
    sqr = torch.pow(diff, 2)
    hell = torch.sqrt(0.5 * torch.sum(sqr, dim=1))
    return hell.cpu().detach().numpy()


class selectiveMisinformation:
    def __init__(
        self, model_mis, delta_list, num_classes=10, rand_fhat=False, use_adaptive=True
    ):
        self.delta_list = delta_list
        self.input_count = 0
        self.ood_count = {}
        self.num_classes = num_classes
        self.correct_class_rank = np.zeros(self.num_classes)
        self.mis_correct_count = 0
        self.hell_dist = {}
        self.max_probs = {}
        self.alpha_vals = {}
        self.reset_stats()
        self.model_mis = model_mis
        self.use_adaptive = use_adaptive
        self.rand_fhat = rand_fhat

    def __call__(self, x, y):
        probs = y  # batch x 10
        delta = self.delta_list
        probs_max, probs_max_index = torch.max(probs, dim=1)  # batch
        batch = probs_max.size(0)
        self.input_count += batch
        y_mis = self.model_mis(x)
        y_mis = F.softmax(y_mis, dim=1)
        probs_mis_max, probs_mis_max_index = torch.max(y_mis, dim=1)  # batch
        self.mis_correct_count += (probs_mis_max_index == probs_max_index).sum().item()
        y_mis_dict = {}
        y_mis = y_mis.detach()
        if self.use_adaptive:
            h = 1 / (1 + torch.exp(-10000 * (delta - probs_max.detach())))
        else:
            h = delta * torch.ones_like(probs_max.detach())

        h = h.unsqueeze(dim=1).float()
        mask_ood = probs_max <= delta
        self.ood_count += np.sum(mask_ood.cpu().detach().numpy())
        y_mis_dict = ((1.0 - h) * y) + (h * y_mis.float())
        probs_mis_max, _ = torch.max(y_mis_dict, dim=1)
        self.max_probs.append(probs_mis_max.cpu().detach().numpy())
        self.alpha_vals.append(h.squeeze(dim=1).cpu().detach().numpy())

        hell = compute_hellinger(y_mis_dict, y)
        self.hell_dist.append(hell)
        return y_mis_dict

    def get_stats(self):
        rejection_ratio = {}
        for delta in self.delta_list:
            rejection_ratio = float(self.ood_count) / float(
                self.input_count
            )
            print("Delta: {} Rejection Ratio: {}".format(delta, rejection_ratio))
            self.hell_dist = np.array(np.concatenate(self.hell_dist))
            self.max_probs = np.array(np.concatenate(self.max_probs))
            self.alpha_vals = np.array(np.concatenate(self.alpha_vals))
        print(
            "miss_correct_ratio: ",
            float(self.mis_correct_count) / float(self.input_count),
        )
        np.savez_compressed("./logs/hell_dist_sm", a=self.hell_dist)
        np.savez_compressed("./logs/max_probs", a=self.max_probs)
        np.savez_compressed("./logs/alpha_vals", a=self.alpha_vals)
        return rejection_ratio

    def reset_stats(self):
        self.ood_count = 0
        self.hell_dist = []
        self.max_probs = []
        self.alpha_vals = []
        self.input_count = 0
        self.mis_correct_count = 0
        self.correct_class_rank = np.zeros(self.num_classes)


class noDefense:
    def __call__(self, x, y):
        y_noDef = {}
        y_noDef[0] = y
        return y_noDef

    def print_stats(self):
        return