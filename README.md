# Unified Mechanism-Specific Amplification by Subsampling and Group Privacy Amplification

<p align="left">
<img src="https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/_my_direct_uploads/group_privacy_thumbnail.png", width="75%">

This is the official implementation of 

["Unified Mechanism-Specific Amplification by Subsampling and Group Privacy Amplification"](https://www.cs.cit.tum.de/daml/group-amplification/)  
Jan Schuchardt, Mihail Stoian*, Arthur Kosmala*, and Stephan Günnemann, NeurIPS 2024.

## Requirements
To install the requirements, execute
```
conda env create -f environment.yaml
```

You also need to download and install the dp_accounting library
```
git clone https://github.com/google/differential-privacy.git
cd differential-privacy
git reset --hard HEAD 0b109e959470c43e9f177d5411603b70a56cdc7a
pip install python/dp_accounting
```

## Installation
You can install this package via `pip install -e .`

## Usage
In order to reproduce all experiments, you will need need to execute the scripts in `seml/scripts` using the config files provided in `seml/configs`.  
We use the [SLURM Experiment Management Library](https://github.com/TUM-DAML/seml), but the scripts are just standard sacred experiments that can also be run without a MongoDB and SLURM installation.  

After computing all results, you can use the notebooks in `plotting` to recreate the figures from the paper.  
In case you do not want to run all experiments yourself, you can just run the notebooks while keeping the flag `overwrite=False` (our results are then loaded from the respective `raw_data` files).

For more details on which config files and plotting notebooks to use for recreating which figure from the paper, please consult [REPROCE.MD](./REPRODUCE.md).

## Cite
Please cite our paper if you use this code in your own work:

```
@InProceedings{Schuchardt2024_Unified,
    author = {Schuchardt, Jan and Stoian, Mihail and Kosmala, Arthur and G{\"u}nnemann, Stephan},
    title = {Unified Mechanism-Specific Amplification by Subsampling and Group Privacy Amplification},
    booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
    year = {2024}
}
```
