# 1.create_patchs_fp：
This script is used to process the case image and divide it into patches. The patch size of the whole slide image scanned by 40x mirror is 512 pixels, and the patch size of the whole slide image scanned by 20x mirror is 256 pixels. The step size is consistent with the patch size.


`python create_patchs_fp --source xxx --save_dir xxx --patch_size 256 --step_size 256 --seg --patch --stitch`

* `--source`: Catalog of whole slide image
* `--save_dir`: Catalog of generated tissue data
* `--patch_size`: Side length of patch
* `--step_size`: Divide the interval of patch
*  `--seg`: Whether to segmentation the tissue area
*  `--patch`: Whether to generate patch
*  `--stitch`: Whether to generate the stitched map of the patch
# 2.tumor_detect_from_tissue.py：
Identify tumor patches in tissue patches.

`python tumor_detect_from_tissue.py --wsi_dir xxx --tissue_file_dir xxx --save_dir xxx --stitch`
* `--wsi_dir`: Catalog of whole slide image
* `--tissue_file_dir`: Catalog of generated tissue data in the previous step
* `--save_dir`: Catalog of generated tumor data 
* `--stitch`: Whether to generate the stitched map of the patch
# 3.extract_features_fp.py：
Generate eigenvector of whole slide image.

`python extract_features_fp.py --data_h5_dir xxx --data_slide_dir xxx --csv_path xxx --feat_dir xxx --batch_size 512 --slide_ext .svs`
* `--data_h5_dir`: Catalog of tumor patch coordinates .h5 files
* `--csv_path`: Catalog of source data attribute genorated in first step
* `--feat_dir`: Catalog of genorated feature
* `--batch_size`: Batch size of generated features
* `--slide_ext`: Whole slide image format
# 4.main_mtl_concat.py：
`python main_mtl_concat.py --lr 1e-4 --k 10 --k_start 0 --k_end 10 --dataset_csv xxx --exp_code xxx --task msi_classifier --log_data --data_root_dir xxx --model_type toad --early_stopping`  
* `--lr`: Learning rate during training
* `--k`: K-fold cross validation  
* `--k_start`: The beginning num of K-fold cross-validation
* `--k_end`: The ending num of K-fold cross-validation
* `--dataset_csv`: Catalog of source data attribute genorated in first step
* `--drop_out`: Whether to drop out
* `--early_stopping`: Whether early stopping
* `--exp_code`: The code for this task
* `--log_data`: Whether to generate logs
* `--data_root_dir`: Catalog of whole slide image
* `--task`: Task name
* `--split_dir` Split dirctory
* `--model_type`: Model type can select toad, toad_cosine, rnn, mil, attmil
# 5.main_cnn_trainer.py：
`python main_cnn_trainer.py --max_epochs 5 --lr 1e-4 --k 10 --k_start 0 --k_end 10 --dataset_csv dataset_csv/xxx.csv --exp_code msi_classifier_vit --task msi_classifier --log_data --data_root_dir xxx/xxx/xxx --model_type vit --split_dir msi_classifier_xxx_100 --batch_size 32 --input_size 384`
* `--lr`: Learning rate during training
* `--data_root_dir` Whole slide image directory
* `--k`: K-fold cross validation  
* `--k_start`: The beginning num of K-fold cross-validation
* `--k_end`: The ending num of K-fold cross-validation
* `--dataset_csv`: Catalog of source data attribute genorated in first step
* `--drop_out`: Whether to drop out
* `--early_stopping`: Whether early stopping
* `--exp_code`: The code for this task
* `--log_data`: Whether to generate logs
* `--data_root_dir`: Catalog of whole slide image
* `--task`: Task name
* `--split_dir` Split dirctory
* `--model_type`: Model type can select resnet18, resnet50, vit, efficient
