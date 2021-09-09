source code：https://github.com/mahmoodlab/CLAM
# 1.create_patchs_fp：
--source  
Catalog of whole slide image  
--save_dir  
Directory of generated data  
--patch_size  
Side length of patch  
--step_size  
Divide the interval of patch  
--seg  
--patch  
--only_tumor  
Use the model in the previous step to generate patch coordinates and patch feature vectors that only contain the tumor area
--normalize
Use staining normalization
# 2.