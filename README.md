# LMC: Homography Matrix-Based Local Motion Consistent Matching for Remote Sensing Images

`LMC.py` is the main program of the LMC method, which includes the implementation of `LMC_LPM()` and `LMC_RANSAC()` functions for LMC_LPM and LMC_RANSAC methods, respectively. `LMC_LPM()` takes putative matches $\{(m1,m2)\}$, $K$, and $\tau$ as input, and the LPM parameters use the default values. `LMC_RANSAC()` takes assumed matches $\{(m1,m2)\}$, $K$, $\tau$, and $\alpha$ as input, and all the parameters have the default values in the paper. If the input is an image instead of putative matches, `SIFT()` function should be used to generate the putative match set.

`Qualitative Analysis.py` is the program that visualizes the matching results of the LMC method in various datasets.

`Test_alg_byH.py` is the program that calculates the performance metrics (F-score, Recall, Precision, and Runtime) of the LMC method in the HPatches dataset.

`Test_alg_byMat.py` is the program that calculates the performance metrics of the LMC method in the RS, DTU, and Retina datasets.

`Test_alg_bySUIRD.py` is the program that calculates the performance metrics of the LMC method in the SUIRD dataset.

`part_of_datasets` contains some of the datasets used in the experiments. However, the DTU, HPatches, and SUIRD datasets are incomplete. For the complete datasets, please visit:

RS, DTU, Retina: https://github.com/StaRainJ/Image_matching_Datasets

HPatches: https://github.com/hpatches/hpatches-dataset  (hpatches-sequences-release)

SUIRD: https://github.com/yyangynu/SUIRD
