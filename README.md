# Poison Can Be Antidote

## Environment
1. PyTorch 1.7.0
2. PuLP 2.6.0

## Instruction

For different datasets, see the corresponding startup codes for detailed setting and running commands. We use CUB200 as the example.

See the file ```CUB200_startup``` for running experments on CUB200.


### General Setup

The following commands run in shell define some common settings.

```shell
################################ CUB200 ################################
### If you have multiple GPUs on the machine, use this to select the specific GPU
dev_id=2
### p_v = victim model dataset
p_v=CUBS200
### f_v = architecture of victim model
f_v=resnet50
### queryset = p_a = image pool of the attacker 
queryset=ImageNet1k
### Path to victim model's directory (the one downloded earlier)
vic_dir=models/victim/${p_v}-${f_v}-train-nodefense
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=50000 
### Batch size of queries to process for the attacker
batch_size=32
### Training parameters
lr=0.01
lr_step=10
lr_gamma=0.5
epochs=30
training_batch_size=32
### pretrained model
pretrained=imagenet
```

```div_id```: the id of device (GPU) to run the experiments.

```p_v```: The training dataset of the victim model.

```f_v```: The model architecture of the victim model.

```queryset```: The query (surrogate) dataset of the shadow model.

```vic_dir```: The path of storing and loading the victim model. It will be generated by training the victim model for the first time.

```budget```: The query budget of the attacker.

```batch_size```: The batch size of query (Notice that it is not the training batch size, see ```training_batch_size``` below).

```lr```: Learning rate of training the shadow model.

```lr_step```: Learning rate decay period of training the shadow model.

```lr_gamma```: Learning rate dacay factor of training the shadow model.

```epochs```: Number of epochs for training the shadow model.

```pretrained```: The pretraining model of both victim model and shadow model. Use imagenet as default.

### Generate Victim Model
After defining the general setup, you can generate the victim model if you have not done this before. Use the following command:

```shell
# (defense) train an original (blackbox) model
python defenses/victim/train.py ${p_v} ${f_v} -o ${vic_dir} -b 64 -d ${dev_id} -e 100 -w 4 --lr 0.01 --lr_step 30 --lr_gamma 0.5 --pretrained ${pretrained}
```

To train an original (blackbox) model with outlier exposure to run the adaptive misinformation attack, use the following command, add argument `--am_flag`:

```shell
# (defense) train an original (blackbox) model with outlier exposure
python defenses/victim/train.py ${p_v} ${f_v} -o ${vic_dir}-OE -b 64 -d ${dev_id} -e 100 -w 4 --lr 0.01 --lr_step 30 --lr_gamma 0.5 --pretrained ${pretrained} --am_flag --dataset_oe ImageNet1k
``` 

You will need to generate victim model twice, once original and once with outlier exposure. These two models will be used for all the experiments on this dataset.

### Define Attack Method
You can define different attacks with the following commands when using different options.

1. KnockoffNet with random selection

```shell
policy=random
hardlabel=0
defense_aware=0
recover_table_size=10000000
recover_norm=1
recover_tolerance=0.01
recover_proc=10
policy_suffix="_da${defense_aware}_hard${hardlabel}"
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};recover_proc:${recover_proc}"
```

```hardlabel```: If set to 1, the attacker only use top-1 label for extraction (Top-1 Attack).

```defense_aware```: If set to 1, the attacker use lookup table recovery (Lookup Table Attack). 

```recover_table_size```: The size of the lookup table.

```recover_norm```: The norm used in lookup table attack. 

```recover_tolerance```: The difference tolerance in which two perturbed labels as treat as one in the lookup table. 

```recover_proc```: Use multiprocess to generate the lookup table since it takes a long time. This is the number of processes.

2. JBDA-based Attack

```shell
## jbda/jbself/jbtop
policy=jbtop3
seedsize=500
jb_epsilon=0.1
T=8
policy_suffix="ss${seedsize}_eps${jb_epsilon}"
```

If you run JBDA, you should use ```jacobian.py``` instead of ```transfer.py``` for getting transfer set.

KnockoffNet and JBDA are exclusive so you only need to use either command block from the above two in your setup. 

3. Semisupervised Attck
Both KnockoffNet and JBDA can be combined with this technique. Use the following command to define the semisupervised loss.

```shell
## semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
```

Notice: Even you do not use the semisupervised attack, you need to set ```semi_train_weight``` to 0 instead of skip this defination.

### Define Defense Method
Use different command blocks for different defense methods. 

1. No Defense
```shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## None
strat=none
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/none/quantize${quantize_epsilon}
# Parameters to defense strategy, provided as a key:value pair string. 
defense_args="out_path:${out_dir}"
```

2. Top-k Defense
```shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Top-k
strat=topk
topk=1
rounding=0
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/top${topk}-rounding${rounding}
defense_args="topk:${topk};rounding:${rounding};out_path:${out_dir}"
```

3. Rounding Defense
```shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## rounding
strat=rounding
rounding=1
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/rounding${rounding}
defense_args="rounding:${rounding};out_path:${out_dir}"
```


4. MAD Defense
```shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## MAD
strat=mad
# Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
ydist=l1
# Perturbation norm
eps=1.0

# version: original MAD
batch_size=1
# Perturbation mode: extreme|argmax
oracle=argmax
# Initialization to the defender's surrogate model. 'scratch' refers to random initialization.
proxystate=scratch
# Path to surrogate model
proxydir=models/victim/${p_v}-${f_v}-train-nodefense-${proxystate}-advproxy
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/mad_${oracle}_${ydist}/eps[${eps},${quantize_epsilon}]-proxy_${proxystate}
# Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps};batch_constraint:0;objmax:1;oracle:${oracle};ydist:${ydist};model_adv_proxy:${proxydir};out_path:${out_dir}"
```

You must generate the proxy model for MAD with the following code before running MAD defense:

```shell
python defenses/victim/train.py ${p_v} ${f_v} --out_path models/victim/${p_v}-${f_v}-train-nodefense-scratch-advproxy --device_id 0 --epochs 1 --train_subset 10 --lr 0.0
```

4. MUAD
``` shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## MAD
strat=mad
# Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
ydist=l1
# Perturbation norm
eps=1.0

# Version: MUAD (Use I as jacobian in MAD)
# Perturbation mode: lp_extreme|lp_argmax
oracle=lp_argmax
# Using Batch Constraint
batch_constraint=0
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/muad_${oracle}_${ydist}/batch${batch_size}-eps[${eps},${quantize_epsilon}]_bc${batch_constraint}
# Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps};batch_constraint:${batch_constraint};objmax:1;oracle:${oracle};ydist:${ydist};disable_jacobian:1;out_path:${out_dir}"
```

6. MKLD
``` shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## MLD
strat=mld
# Using Batch Constraint
batch_constraint=0
# Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
ydist=l1
# Perturbation norm
eps=1.0
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/mld_${ydist}/batch${batch_size}-eps[${eps},${quantize_epsilon}]_bc${batch_constraint}
# Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps};batch_constraint:${batch_constraint};ydist:${ydist};out_path:${out_dir}"

```
7. Adaptive Misinformation
```shell
## Quantization
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

#AM
strat=am
defense_level=0.99

# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/am/def_level${defense_level}

```

8. Ordered Quantization
```shell
### Defense strategy
## Quantization
quantize=1
quantize_epsilon=1.0
optim=0
ydist=l1

# get centroids from query
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## None
strat=none
# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/none/quantize${quantize_epsilon}
# Parameters to defense strategy, provided as a key:value pair string. 
defense_args="out_path:${out_dir}"
```



### Query and Training
After defining attack and defense, you can use the following commands to query and train the shadow model.

1. KnockoffNet
```shell
# (adversary) generate transfer dataset (only when policy=random)
python defenses/adversary/transfer.py ${policy} ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id} --queryset ${queryset} --budget ${budget} --quantize ${quantize} --quantize_args ${quantize_args} --defense_aware ${defense_aware} --recover_args ${recover_params} --hardlabel ${hardlabel}

# (adversary) train kickoffnet and evaluate
python defenses/adversary/train.py ${out_dir} ${f_v} ${p_v} --budgets 50000 -e ${epochs} -b ${training_batch_size} --lr ${lr} --lr_step ${lr_step} --lr_gamma ${lr_gamma} -d ${dev_id} -w 4 --pretrained ${pretrained} --vic_dir ${vic_dir} --semitrainweight ${semi_train_weight} --semidataset ${semi_dataset} 
```
\

2. JBDA-based
```shell
# (adversary) use jbda/jbda-topk as attack policy
python defenses/adversary/jacobian.py ${policy} ${vic_dir} ${strat} ${defense_args} --model_adv ${f_v} --pretrained imagenet --out_dir ${out_dir} --testset ${p_v} --batch_size 128 -d ${dev_id} --queryset ${queryset} --budget ${budget} --seedsize ${seedsize} --epsilon ${jb_epsilon} --T ${T} --train_epochs 20
```
