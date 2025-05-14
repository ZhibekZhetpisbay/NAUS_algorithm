# NAUS
This repository contains an implementation of Noiseware-Aware Undersampling with Subsampling (NAUS) designed to address class imbalance in datasets, where the majority class significantly outnumbers the minority class.
## How the algorithm works?
This algorithm is designed to improve classification performance on imbalanced datasets through a two-stage undersampling strategy, preceded by noise filtering based on posterior probability estimation.
**Step 1: Noise Removal via Posterior Probability**
    Initially, the algorithm estimates the posterior probability of each sample belonging to its labeled class.
  Samples with low posterior probability are considered noisy or mislabeled and are removed from the dataset. This enhances the quality of the remaining data before     undersampling is applied.

**Step 2: Two-Stage Undersampling**
  The undersampling procedure consists of two adaptive stages:

  • Stage 1: Tomek Link Removal and Pre-sampling
  
    - Tomek links are identified and removed to eliminate borderline majority-class instances that cause class overlap.
    - The dataset is then split into training and evaluation subsets.
    - If the resulting class distribution meets the predefined ratio threshold, the process stops here.
  
  • Stage 2: Potential-Based Undersampling
    If class imbalance persists, the algorithm performs a second undersampling phase by evaluating the class potential of each instance:
    
    - Class potential is computed based on local data density and class distribution.
    - Majority-class samples with low contribution to classification boundary are selectively removed, 
      improving class balance without discarding informative samples.
  ## Visualized results
<img src="https://github.com/user-attachments/assets/e06720c1-d1ca-4c7f-8bfb-f2ec09261145" width="500"/>
<img src="https://github.com/user-attachments/assets/688e280b-0442-49a5-a7e5-4cebdb728553" width="500"/>


## How to Use?
The implementation of the NAUS algorithm and accompanying training procedures is publicly available in this [repository](NAUS_code).
Comprehensive usage examples and step-by-step instructions are provided for the benchmark datasets us_crime and sat_image, located in the [repository](NAUS_tutorials).
These examples demonstrate how to apply the algorithm from data preprocessing to model training and evaluation.
> ⚠️ **Before using the package**, please make sure to update all required libraries to their latest stable versions to avoid compatibility issues.  
>  
> ✅ Below is an example of tested library versions known to work properly with this package:
>
> - `pip` 23.2.1  
> - `imbalanced-learn` 0.12.3  
> - `lightgbm` 4.5.0  
> - `matplotlib` 3.9.2  
> - `numpy` 1.26.4  
> - `pandas` 2.2.3  
> - `scikit-learn` 1.5.1  
> - `scipy` 1.11.1  
> - `torch` 2.5.1  
> - `umap-learn` 0.5.6  
> - `xgboost` 2.1.4  
> - `setuptools` 75.3.0  
> - `packaging` 25.0  
