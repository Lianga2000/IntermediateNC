# Comparative Generalization Bounds for Deep Neural Networks

## Requirements
- Python 3.10
- Pytorch 1.11
- Numpy
- Tqdm

## Running Experiments

**To submit the code as a job to slurm:**
    ```
    sh train.sh
    ```

Other Files

* conf/global_settings.py: A file that specifies the configuration parameters and hyperparameters.
* analysis.py: Contains functions that help in measuring cdnv (class-distance normalized variance as defined in the paper), accurecies, losses, the level of nearest class center rule.
* utils.py: Contains functions responsible for saving data, loading datasets, etc.
* models: Contains implementations of networks used in training.

<br />
<hr> 
<h3> Citation </h3>

```bib
@article{galanti2023comparative,
  title={Comparative generalization bounds for deep neural networks},
  author={Galanti, Tomer and Galanti, Liane and Ben-Shaul, Ido},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```
