# Counterfactual Identifiability of Bijective Causal Models

We study counterfactual identifiability in causal models with bijective generation mechanisms (BGM), a class that generalizes several widely used causal models in the literature. 
We establish their counterfactual identifiability for three common causal structures with unobserved confounding, 
and propose a practical learning method that casts learning a BGM as structured generative modeling. 
Learned BGMs enable efficient counterfactual estimation and can be obtained using a variety of deep conditional generative models. 
We evaluate our techniques in a visual task and demonstrate its application in a real-world video streaming simulation task.

***
## Requirements
We use Python (3.9.11 tested) for all experiments. Install the following packages:
```
numpy, tqdm, matplotlib, pyro
```
Install PyTorch according to the website [instructions](https://pytorch.org).

---
## Counterfactual Ellipse Generation
Create a directory as a workspace for datasets, models and plots. We'll call this directory `DIR`.
```
python ellipse/generate_data.py --dir DIR
```
generates the training data.
To train BC and baseline models use the following scripts:
```
python ellipse/train.py --dir DIR
python ellipse/baseline_x_train.py --dir DIR
python ellipse/baseline_xz_train.py --dir DIR
```
Table 1's numbers are taken from logs in `DIR/logs`.
Finally, run the following command to generate Figure 4 `(DIR/ellipse.png)`:
```
python ellipse/generate_ellipse.py --dir DIR
```
---
## Video Streaming Simulation
Create a directory as a workspace for datasets, models and plots. We'll call this directory `DIR`.
Move `video/simulator_data.tar.gz` (generated with [this code](https://github.com/CausalSim/Unbiased-Trace-Driven-Simulation/blob/master/abr-synthetic/create_dataset_and_expertsim.py) to `DIR` and extract it using the following command:
```
tar -xzvf DIR/simulator_data.tar.gz
```
For all training runs, we tried seeds (`SEED`) from 0 to 9.
### Markovian
Create the training dataset by running the following command:
```
python video/create_markovian_data.py --dir DIR
```
Next, train the model using the following script:
```
python video/train_markovian.py --dir DIR --seed SEED
```
The `Markovian` row in Table 1 was generated using logs in `DIR/markovian/logs`.
### Instrumental Variable (IV)
Create the training dataset by running the following command:
```
python video/create_IV_data.py --dir DIR
```
Next, train the `IV` and `CausalSim` models using the following scripts:
```
python video/train_IV.py --dir DIR --seed SEED
python video/train_causalsim.py --dir DIR --seed SEED
```
The `IV` and `CausalSim` rows in Table 1 was generated using logs in `DIR/iv/logs`.

### Backdoor Criterion (BC)
Create the training dataset by running the following command:
```
python video/create_BC_data.py --dir DIR
```
Next, train models for both buffer based ABR algorithms (`bba` and `bola`) using the following scripts:
```
python video/train_bc.py --dir DIR --seed SEED --algo bba
python video/train_bc.py --dir DIR --seed SEED --algo bola
```
The `BC` row in Table 1 was generated using logs in `DIR/bc/logs`, by taking the average over `Bola` and `BBA` algorithms.
### IV + BC
Train the model using the following script:
```
python video/train_iv_bc_mix.py --dir DIR --seed SEED
```
The `IV+BC` row in Table 1 was generated using logs in `DIR/mix/logs`.