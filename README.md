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
  
