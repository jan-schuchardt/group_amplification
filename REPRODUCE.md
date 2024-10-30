# Reproducing our experiments

In the following, we describe which scripts and notebooks to run in which order to reproduce the different figures from our paper.

Note that you may have to adjust the directories in the individual config files (to point at the correct dataset folders, result folders etc.).  
You have to manually create these directories, they are not automatically created by the program itself.  
You will also need to adjust the slurm configuration parameters at the top of each file to match your cluster configuration (partition names etc.).

If you do not want to train and certify the models yourself, you can just run the plotting notebooks while keeping the flag `overwrite=False`.  
If you do, you will need to set `overwrite=True` when running the notebook for the first time.

## Mechanism-specific and mechanism-agnostic bounds 

### Randomized response and RDP - Figs. 3, 7
```
seml group_amplification_neurips24_rdp add seml/configs/rdp/without_replacement/logspace/baseline/baseline_randomized_response_self_consistency.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/without_replacement/logspace/tight/tight_randomized_response.yaml start
```
Then run `plotting/camera/rdp/without_replacement/specific_vs_agnostic/randomized_response_half_page.ipynb`.

### Group Privacy and ADP (Gaussian) - Fig. 8
```
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/maximal_coupling_gaussian.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/maximal_coupling_gaussian.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/tight_gaussian.yaml start

```
Then run `plotting/camera/adp/poisson/specific_vs_agnostic_group/gaussian_half_page.ipynb`.

### Group Privacy and ADP (Laplace) - Figs. 4, 9
```
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/maximal_coupling_laplace.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/tight_laplace.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/maximal_coupling_laplace.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/tight_laplace.yaml start

```
Then run `plotting/camera/adp/poisson/specific_vs_agnostic_group/laplace_half_page.ipynb`.

### Group Privacy and ADP (Randomized response) - Fig. 10
```
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/maximal_coupling_randomized_response.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/tight_randomized_response.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/maximal_coupling_randomized_response.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/tight_randomized_response.yaml start

```
Then run `plotting/camera/adp/poisson/specific_vs_agnostic_group/randomized_response_half_page.ipynb`.

### Group Privacy and RDP - Fig. 11
```
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/maximal_coupling_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/maximal_coupling_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/maximal_coupling_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/tight_gaussian.yaml start
```
Then run `plotting/camera/rdp/poisson/specific_vs_agnostic_group/gaussian_half_page.ipynb`.

### Benefit of Conditioning - Fig. 12
```
seml group_amplification_neurips24_rdp add seml/configs/rdp/without_replacement/logspace/baseline/baseline_gaussian_self_consistency.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/without_replacement/logspace/direct_transport/direct_transport_gaussian.yaml start
```
Then run `plotting/camera/rdp/without_replacement/direct_transport_vs_posthoc_group/gaussian_half_page.ipynb`.

### Subsampling with replacement - Fig. 13
Run `plotting/camera/adp/with_replacement/batch_substitution.ipynb`.

## Post-hoc and tight group privacy

### Single-iteration ADP (Gaussian) - Figs. 5, 14
```
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/baseline_gaussian.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/baseline_gaussian.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/tight_gaussian.yaml start

```
Then run `plotting/camera/adp/poisson/specific_vs_posthoc_group/gaussian_half_page.ipynb`.

### Single-iteration ADP (Laplace) - Fig. 15
```
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/baseline_laplace.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/tight_laplace.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/baseline_laplace.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/tight_laplace.yaml start

```
Then run `plotting/camera/adp/poisson/specific_vs_posthoc_group/laplace_half_page.ipynb`.

### Single-iteration ADP (Randomized response) - Fig. 16
```
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/baseline_randomized_response.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/linspace_continuous/tight_randomized_response.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/baseline_randomized_response.yaml start
seml group_amplification_neurips24_adp add seml/configs/adp/poisson/logspace_continuous/tight_randomized_response.yaml start

```
Then run `plotting/camera/adp/poisson/specific_vs_posthoc_group/randomized_response_half_page.ipynb`.

### Single-iteration RDP - Fig. 17
```
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/baseline_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/baseline_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/baseline_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/tight_gaussian.yaml start
```
Then run `plotting/camera/rdp/poisson/specific_vs_posthoc_group/gaussian_half_page.ipynb`.

### PLD Accounting (Gaussian) - Figs. 6, 18, 19
```
seml group_amplification_neurips24_pld add seml/configs/composition/pld/baseline_gaussian.yaml start
seml group_amplification_neurips24_pld add seml/configs/composition/pld/tight_gaussian.yaml start
```
Then run `plotting/camera/pld/poisson/specific_vs_posthoc_group_accounting/gaussian_half_page.ipynb`.

### PLD Accounting (Laplace) - Figs. 20, 21
```
seml group_amplification_neurips24_pld add seml/configs/composition/pld/baseline_laplace.yaml start
seml group_amplification_neurips24_pld add seml/configs/composition/pld/tight_laplace.yaml start
```
Then run `plotting/camera/pld/poisson/specific_vs_posthoc_group_accounting/laplace_half_page.ipynb`.

### RDP Accounting (Gaussian) - Fig. 22
```
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/baseline_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/baseline_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/tight_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/baseline_gaussian.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/tight_gaussian.yaml start
```
Then run `plotting/camera/rdp/poisson/specific_vs_posthoc_group_accounting/gaussian_half_page.ipynb`.

### RDP Accounting (Randomized response) - Fig. 23
```
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/baseline_randomized_response.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/linspace_continuous/tight_randomized_response.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/baseline_randomized_response.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace_continuous/tight_randomized_response.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/baseline_randomized_response.yaml start
seml group_amplification_neurips24_rdp add seml/configs/rdp/poisson/logspace/tight_randomized_response.yaml start
```
Then run `plotting/camera/rdp/poisson/specific_vs_posthoc_group_accounting/randomized_response_half_page.ipynb`.

### Model utility - Fig. 24
Run `plotting/camera/mnist/tight_vs_traditional.ipynb`.