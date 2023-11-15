# ModelGuard: Information-Theoretic Defense Against Model Extraction Attacks

This is an official implementation of the USENIX Security 2024 paper *ModelGuard: Information-Theoretic Defense Against Model Extraction Attacks.*

**Abstract**: Malicious utilization of a query interface can compromise the confidentiality of ML-as-a-Service (MLaaS) systems via model extraction attacks. Previous studies have proposed to perturb the predictions of the MLaaS system as a defense against model extraction attacks. However, existing prediction perturbation methods suffer from a poor privacy-utility balance and cannot effectively defend against the latest adaptive model extraction attacks. In this paper, we propose a novel prediction perturbation defense named ModelGuard, which aims at defending against adaptive model extraction attacks while maintaining a high utility of the protected system. We develop a general optimization problem that considers different kinds of model extraction attacks, and ModelGuard provides an information-theoretic defense to efficiently solve the optimization problem and achieve resistance against adaptive attacks. Experiments show that ModelGuard attains significantly better defensive performance against adaptive attacks with less loss of utility compared to previous defenses.

## Environment
1. Python 3.8
2. PyTorch 1.7.1
3. PuLP 2.6.0
4. CUDA >= 10.2

You can run the following commands to create a new environments for running the codes with Anaconda:
```shell
conda env create -f environment.yml
conda activate modelguard
```

**Notice:** Different GPUs may require different versions of PyTorch. Please follow the instructions on the [official website of PyTorch](https://pytorch.org/get-started/locally/) if there is any problem with installing PyTorch. 


## Dataset Preparing
We will use 8 datasets in total in our experiments. Three datasets (CIFAR100, CIFAR10, SVHN) can be automatically downloaded when executing scripts. However, you still need to download **all** the following datasets into ```./data/``` (create it if it does not exist) and unzip them before running any codes. (You can change the default dataset path by changing the ```root``` parameter in the dataset files such as ```./defenses/datasets/caltech256.py```.)

1. [Caltech256](https://data.caltech.edu/records/nyy15-4j048)
2. [CUB200](https://data.caltech.edu/records/65de6-vp158)
3. [ImageNet1k (ILSVRC2012)](http://image-net.org/download-images)
4. [TinyImageNet200](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
5. [Indoor67](http://web.mit.edu/torralba/www/indoor.html)

For Caltech256, CUB200, TinyImageNedt200, Indoor67 and ImageNet1k, you can also use the scripts ```dataset.sh``` to download and unzip in shell:
```shell
sh dataset.sh
```

## Instructions to Run the Codes

We provide automatic scripts to generate the results in our paper. For instance, after preparing all datasets, using the following command will run all the experiments in Table 2. 

```shell
python scripts/run_caltech256.py
```

However, we strongly encourage you to explore more customized running options as instructed below. We will show how to run the experiments with $X_t$ = Caltech256 as an example here. To run the experiments on other dataset, please change the parameters according to our paper.

### 1. General Setup

Run the following commands in shell to define the datasets and training hyperparameters.

```shell
# We currently only support single GPU usage. You can use this to specify which GPU you want to use
dev_id=0
# X_t = target model dataset (Caltech256, CUBS200, CIFAR100, CIFAR10)
X_t=Caltech256
# w_t = architecture of target model (resnet50, vgg16_bn)
w_t=resnet50
# queryset = X_q = image pool of the attacker (ImageNet1k, TinyImageNet200)
queryset=ImageNet1k
# oeset = X_{OE} = outlier exposure dataset (Indoor67, SVHN)
oeset=Indoor67
# lambda of OE
oe_lamb=1.0
# Path to target model's directory
vic_dir=models/victim/${X_t}-${w_t}-train-nodefense
# Use pretrained model for initialization of both target model and substitute model
pretrained=imagenet
# Training parameters for the substitute model
lr=0.01
lr_step=10
lr_gamma=0.5
epochs=30
training_batch_size=32
```

You need to generate the target model if you do not have the target model. Run the following command in shell to train the model with outlier exposure:

```shell
# (defense) Train a target model with outlier exposure
python defenses/victim/train_admis.py ${X_t} ${w_t} -o ${vic_dir} -b 64 -d ${dev_id} -e 100 -w 4 --lr 0.01 --lr_step 30 --lr_gamma 0.5 --pretrained ${pretrained} --oe_lamb ${oe_lamb} -doe ${oeset}
```

This command will also generate the misinformation model for AM defense.


### 2. Define Attack Method
You can specify the query strategy (KnockoffNet or JBDA-TR) for the attacker from the following two options by running the corresponding command lines in shell.

1. KnockoffNet

    ```shell
    policy=random
    budget=50000 
    # Batch size of queries to process for the attacker. Set to 1 to simulate the realtime sequential query.
    batch_size=1
    ```

2. JBDA-TR

    ```shell
    policy=jbtr3
    budget=50000 
    seedsize=1000
    jb_epsilon=0.1
    T=8
    # Batch size of queries to process for the attacker. Set to 1 to simulate the realtime sequential query.
    batch_size=1
    ```

Then you need to choose one of the following attack strategies by running the corresponding command lines in shell:

#### a. Naive Attack

```shell
## Adaptive setting
hardlabel=0
defense_aware=0
recover_table_size=1000000
recover_norm=1
recover_tolerance=0.01
concentration_factor=8.0
recover_proc=5
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};concentration_factor:${concentration_factor};recover_proc:${recover_proc}"
## Semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
## Augmentation setting
transform=0
# Set qpi=2 to run 2-QPI Attack
qpi=1
policy_suffix="_naive"
```

#### b. Top-1 Attack

```shell
## Adaptive setting
hardlabel=1
defense_aware=0
recover_table_size=1000000
recover_norm=1
recover_tolerance=0.01
concentration_factor=8.0
recover_proc=5
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};concentration_factor:${concentration_factor};recover_proc:${recover_proc}"
## Semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
## Augmentation setting
transform=0
qpi=1
policy_suffix="_top1"
```

#### c. S4L Attck

```shell
## Adaptive setting
hardlabel=0
defense_aware=0
recover_table_size=1000000
recover_norm=1
recover_tolerance=0.01
concentration_factor=8.0
recover_proc=5
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};concentration_factor:${concentration_factor};recover_proc:${recover_proc}"
## Semisupervised setting
semi_train_weight=1.0
semi_dataset=${queryset}
## Augmentation setting
transform=0
qpi=1
policy_suffix="_s4l"
```

**Notice:** S4L Attack can only be used with KnockoffNet Currently.

#### d. Smoothing Attack

```shell
## Adaptive setting
hardlabel=0
defense_aware=0
recover_table_size=1000000
recover_norm=1
recover_tolerance=0.01
concentration_factor=8.0
recover_proc=5
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};concentration_factor:${concentration_factor};recover_proc:${recover_proc}"
## Semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
## Augmentation setting
transform=1
qpi=3
policy_suffix="_smoothing"
```

**Notice:** Smoothing Attack can only be used with KnockoffNet Currently.

#### e. D-DAE

```shell
## Adaptive setting
hardlabel=0
defense_aware=1
shadow_model=alexnet
num_shadows=20
shadowset=ImageNet1k
num_classes=256
shadow_path=models/victim/${X_t}-${shadow_model}-shadow
recover_table_size=1000000
# Use multiprocessing for generating training set. Set to 1 to disable multiprocessing.
recover_proc=5
recover_params="table_size:${recover_table_size};shadow_path:${shadow_path};recover_proc:${recover_proc};recover_nn:1"
## Semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
## Augmentation setting
transform=0
qpi=1
policy_suffix="_ddae"
```

**Notice**: If you have not generate shadow models for generating the training data for D-DAE, you need to run the following command to train the shadow models:

```shell
# (adversarial) Train shadow models
python defenses/victim/train_shadow.py ${shadowset} ${shadow_model} -o ${shadow_path} -b 64 -d ${dev_id} -e 5 -w 4 --lr 0.01 --lr_step 3 --lr_gamma 0.5 --pretrained ${pretrained} --num_shadows ${num_shadows} --num_classes ${num_classes}
```

#### f. D-DAE+

```shell
## Adaptive setting
hardlabel=0
defense_aware=1
recover_table_size=1000000
concentration_factor=8.0
# Use multiprocessing for generating lookup table. Set to 1 to disable multiprocessing.
recover_proc=5
recover_params="table_size:${recover_table_size};concentration_factor:${concentration_factor};recover_proc:${recover_proc};recover_nn:1"
## Semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
## Augmentation setting
transform=0
qpi=1
policy_suffix="_ddae+"
```

**Notice**: You do not need to train shadow models for D-DAE+.

#### g. Partial Bayes Attack

```shell
## Adaptive setting
hardlabel=0
defense_aware=1
recover_table_size=1000000
recover_norm=1
recover_tolerance=0.01
concentration_factor=8.0
# Use multiprocessing for generating lookup table. Set to 1 to disable multiprocessing.
recover_proc=5
recover_params="table_size:${recover_table_size};recover_norm:${recover_norm};tolerance:${recover_tolerance};concentration_factor:${concentration_factor};recover_proc:${recover_proc}"
## Semisupervised setting
semi_train_weight=0.0
semi_dataset=${queryset}
## Augmentation setting
transform=0
qpi=1
policy_suffix="_bayes"
```

### 3. Define Defense Method
You can select one of the following defense against model extraction attacks by running the corresponding command lines in shell. 

#### a. No Defense (None)

```shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=none
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/none
defense_args="out_path:${out_dir}"
```

#### b. Reverse Sigmoid (RevSig)

```shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=reverse_sigmoid
beta=0.008
gamma=0.2
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/revsig/beta${beta}-gamma${gamma}
defense_args="beta:${beta};gamma:${gamma};out_path:${out_dir}"
```


#### c. Maximizing Angular Deviation (MAD)

```shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=mad
ydist=l1
# Set the epsilon here
eps=1.0
oracle=argmax
proxystate=scratch
proxydir=models/victim/${X_t}-${w_t}-train-nodefense-${proxystate}-advproxy
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/mad_${oracle}_${ydist}/eps${eps}-proxy_${proxystate}
defense_args="epsilon:${eps};batch_constraint:0;objmax:1;oracle:${oracle};ydist:${ydist};model_adv_proxy:${proxydir};out_path:${out_dir}"
```

**Notice**: To run MAD, you need to first generate a surrogate model. Run the following command to do so if you have not done this before:

```shell
# (defense) Generate randomly initialized surrogate model
python defenses/victim/train.py ${X_t} ${w_t} --out_path ${proxydir} --device_id ${dev_id} --epochs 1 --train_subset 10 --lr 0.0
```

#### d. Adaptive Misinformation (AM)

The misinformation model for AM is trained automatically when you train the target model with OE before.

```shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=am
# Set the tau here
defense_lv=0.25
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/am/tau${defense_lv}
defense_args="defense_level:${defense_lv};out_path:${out_dir}"
```

#### e. Top-1 Defense

```shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=topk
topk=1
rounding=0
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/top${topk}
defense_args="topk:${topk};rounding:${rounding};out_path:${out_dir}"
```

#### f. Rounding Defense

```shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=rounding
rounding=1
# Output path to attacker's model
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/rounding${rounding}
defense_args="rounding:${rounding};out_path:${out_dir}"
```


#### g. ModelGuard-W

``` shell
## Quantization setting
quantize=0
quantize_epsilon=0.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=mld
batch_constraint=0
ydist=l1
# Set the epsilon here
eps=1.0
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/modelguardw/eps${eps}
defense_args="epsilon:${eps};batch_constraint:${batch_constraint};ydist:${ydist};out_path:${out_dir}"
```

#### h. ModelGuard-S

```shell
## Quantization setting
quantize=1
# Set the epsilon here
quantize_epsilon=1.0
optim=0
ydist=l1
frozen=0
quantize_args="epsilon:${quantize_epsilon};ydist:${ydist};optim:${optim};frozen:${frozen};ordered_quantization:1"

## Perturbation setting
strat=none
out_dir=models/final_bb_dist/${X_t}-${w_t}/${policy}${policy_suffix}-${queryset}-B${budget}/modelguards/eps${quantize_epsilon}
defense_args="out_path:${out_dir}"
```

#### Evaluate the Protected Model
You can get the protected accuracy of the target model defended by the specified defense method by running the following command:

```shell
python defenses/victim/eval.py ${vic_dir} ${strat} ${defense_args} --quantize ${quantize} --quantize_args ${quantize_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id}
```

### 4. Query and Training
After defining attack and defense, you can use the following commands to query and train the substitute model based on the query strategy you use.

1. KnockoffNet

    **Query**:

    ```shell
    # (adversary) Generate transfer dataset (only when policy=random)
    python defenses/adversary/transfer.py ${policy} ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id} --queryset ${queryset} --budget ${budget} --quantize ${quantize} --quantize_args ${quantize_args} --defense_aware ${defense_aware} --recover_args ${recover_params} --hardlabel ${hardlabel} --train_transform ${transform} --qpi ${qpi}
    ```

    **Train**:

    ```shell
    # (adversary) Train kickoffnet and evaluate
    python defenses/adversary/train.py ${out_dir} ${w_t} ${X_t} --budgets 50000 -e ${epochs} -b ${training_batch_size} --lr ${lr} --lr_step ${lr_step} --lr_gamma ${lr_gamma} -d ${dev_id} -w 4 --pretrained ${pretrained} --vic_dir ${vic_dir} --semitrainweight ${semi_train_weight} --semidataset ${semi_dataset} 
    ```

2. JBDA-TR

    ```shell
    # (adversary) Use jbda-tr as attack policy
    python defenses/adversary/jacobian.py ${policy} ${vic_dir} ${strat} ${defense_args} --quantize ${quantize} --quantize_args ${quantize_args} --defense_aware ${defense_aware} --recover_args ${recover_params} --hardlabel ${hardlabel} --model_adv ${w_t} --pretrained ${pretrained} --out_dir ${out_dir} --testdataset ${X_t} -d ${dev_id} --queryset ${queryset} --query_batch_size ${batch_size} --budget ${budget} -e ${epochs} -b ${training_batch_size} --lr ${lr} --lr_step ${lr_step} --lr_gamma ${lr_gamma} --seedsize ${seedsize} --epsilon ${jb_epsilon} --T ${T} 
    ```

You will get the trained substitute model and training logs in ```out_dir```.