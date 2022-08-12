import os
################################ CIFAR-10 ################################
### If you have multiple GPUs on the machine, use this to select the specific GPU
dev_id=6
### p_v = victim model dataset
p_v="CIFAR10"
### f_v = architecture of victim model
f_v="vgg16_bn"
### queryset = p_a = image pool of the attacker 
queryset="CIFAR100"
### Path to victim model's directory (the one downloded earlier)
vic_dir="models/victim/{}-{}-train-nodefense".format(p_v,f_v)
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=50000 
### Batch size of queries to process for the attacker
batch_size=32

### attack policy
## pretrained model
pretrained=vic_dir

## random
policy="random"
defense_aware=0
policy_suffix="_da{}".format(defense_aware)

### Defense strategy
## MAD
strat="mad"
# Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
ydist="l1"
# Perturbation mode: extreme|argmax|lp_extreme|lp_argmax
oracle="lp_argmax"
# Using Batch Constraint
batch_constraint=0
# Perturbation norm
eps= [0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
for e in eps:
    # MAD: Use I as jacobian
    # Output path to attacker's model
    out_dir="models/final_bb_dist/{}-{}-mad_{}_{}_batch{}-eps{}_bc{}-{}-B{}-noproxy-{}{}".format(
        p_v,f_v,oracle,ydist,batch_size,e,batch_constraint,queryset,budget,policy,policy_suffix
    )
    # Parameters to defense strategy, provided as a key:value pair string. 
    defense_args="epsilon:{},batch_constraint:{},objmax:1,oracle:{},ydist:{},disable_jacobian:1,out_path:{}".format(
        e,batch_constraint,oracle,ydist,out_dir
    )

    # (adversary) generate transfer dataset (only when policy=random)
    command_transfer = "python defenses/adversary/transfer.py {} {} {} {} --out_dir {} --batch_size {} \
                        -d {} --queryset {} --budget {} --defense_aware {}".format(
                            policy,vic_dir,strat,defense_args,out_dir,batch_size,dev_id,queryset,budget,defense_aware
                            )
    # (adversary) train kickoffnet and evaluate
    command_train = "python defenses/adversary/train.py {} {} {} --budgets 50000 -e 50 -b 128 -d {} -w 4 --pretrained {} --vic_dir {} --fix_feat".format(
                        out_dir,f_v,p_v,dev_id,pretrained,vic_dir)
    
    if not os.path.exists(out_dir+"/transferset.pickle"):
        os.system(command_transfer)
    os.system(command_train)