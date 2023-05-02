```shell
### VICTIM SETTINGS
### dev_id = GPU device ID
dev_id=0
### p_v = victim model dataset
p_v=Caltech256
### f_v = architecture of victim model
f_v=resnet50
### queryset = p_a = image pool of the attacker 
queryset=ImageNet1k
### train oe model as well if we want to run adaptive misinformation attack
am_flag=True
oeset=Indoor67

if [ ${am_flag} = True ]; then
    am_suffix="-OE-${oeset}"
else
    am_suffix=""
fi

### Path to victim model's directory (the one downloded earlier)
vic_dir=models/victim/${p_v}-${f_v}-train-nodefense${am_suffix}
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


### TRAIN VANILLA VICTIM MODEL

python defenses/victim/train.py ${p_v} ${f_v} -o ${vic_dir} -b 64 -d ${dev_id} -e 100 -w 4 --lr 0.01 --lr_step 30 --lr_gamma 0.5 --pretrained ${pretrained}

### TRAIN OE AND POISON MODEL
python defenses/victim/train_admis.py ${p_v} ${f_v} -o ${vic_dir} -b 32 -d ${dev_id} -e 100 -w 4 --lr 0.01 --lr_step 30 --lr_gamma 0.5 --pretrained ${pretrained}

### KnockoffNets settings
policy=random
hardlabel=0
defense_aware=0
recover_table_size=10000000
recover_norm=1
recover_tolerance=0.01
recover_proc=10
policy_suffix="_da${defense_aware}_hard${hardlabel}"
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};recover_proc:${recover_proc}"

### Defense settings
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
# proxydir=models/victim/${p_v}-${f_v}-train-nodefense-${proxystate}-advproxy

# Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}/${policy}${policy_suffix}-${queryset}-B${budget}/${strat}/def_level${defense_level}_${oeset}
# Parameters to defense strategy, provided as a key:value pair string. 
defense_args="defense_level:${defense_level};out_path:${out_dir};"


### TRANSFER
python defenses/adversary/transfer.py ${policy} ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id} --queryset ${queryset} --budget ${budget} --quantize ${quantize} --quantize_args ${quantize_args} --defense_aware ${defense_aware} --recover_args ${recover_params} --hardlabel ${hardlabel}


### TRAIN ATTACKER AND EVALUATE
python defenses/adversary/train.py ${out_dir} ${f_v} ${p_v} --budgets 50000 -e ${epochs} -b ${training_batch_size} --lr ${lr} --lr_step ${lr_step} --lr_gamma ${lr_gamma} -d ${dev_id} -w 4 --pretrained ${pretrained} --vic_dir ${vic_dir} 

```