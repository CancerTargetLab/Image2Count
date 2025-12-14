Download [data](https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/nsclc-ffpe-dataset/) from nanostring:
```sh
mdkir data/raw/cosmx/raw/
cd data/raw/cosmx/raw/
wget https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/All+SMI+data.tar.gz
tar -xvzf All+SMI+data.tar.gz
rm All+SMI+data.tar.gz
```

Next we remove some image files that are either from an unsuccessful CosMx run, or do not have cell positions/transcript counts:
```sh
rm Lung9_Rep2/*Raw*/20210811*
rm */*/20210907_180607_S2_C902_P99_N99_F031_Z00*.TIF
rm */*/20210907_180607_S2_C902_P99_N99_F032_Z00*.TIF
rm Lung5*/*/*F031_Z00*.TIF
rm Lung5*/*/*F032_Z00*.TIF
rm -r Lung5_Rep2/*Raw*/20210823_175820_S1_C902_P99_N99_F006_Z00*
```

We want to now create the files necessary to run Image2Count. For that we modify `src/utils/prepare_cosmx.py` in line 5-10:
```py
tif_files = glob.glob(' */*Raw*/*_Z001.TIF')
exprMat_files = glob.glob('*/*Flat_files_and_images/*exprMat_file*')
metdaData_files = glob.glob('*/*Flat_files_and_images/*metadata_file*')

experiment_name_path_idx = 0
ignore_cell_id = [0]
```
and `src/utils/cosmx_label_and_pos_data.py` in line 5 to 7 to:
```py
tif_files = glob.glob('*/*Raw*/*_Z001.TIF')

experiment_name_path_idx = 0
```
and line 18 to:
```py
    tif_dict[file.split('/')[experiment_name_path_idx]][int(file.split('/')[-1].split('.')[0].split('F')[-1].split('_')[0])] = file.split('/')[-1]
```

We can now create `cosmx_measurements.csv` and `cosmx_label.csv`(flipping y axis to fit our implementation) and create the functioning experiment folder utilizing images from only the first layer of the Z-Stack:
```sh
python ../../../../src/utils/prepare_cosmx.py
python ../../../../src/utils/cosmx_label_and_pos_data.py
python ../../../../src/utils/flip_y_axis.py
cd ../../../../
mkdir data/raw/cosmx/train
mdkir data/raw/cosmx/test
ln -s path/to/data/raw/cosmx/raw/cosmx_measurements_flipped_y.csv path/to/data/raw/cosmx/
ln -s path/to/data/raw/cosmx/raw/cosmx_measurements_flipped_y.csv path/to/data/raw/
ln -s path/to/data/raw/cosmx/raw/cosmx_label.csv path/to/data/raw/cosmx/
ln -s path/to/data/raw/cosmx/raw/Lung6/*/*_Z001.TIF path/to/data/raw/cosmx/test/
ln -s path/to/data/raw/cosmx/raw/Lung13/*/*_Z001.TIF path/to/data/raw/cosmx/test/
ln -s path/to/data/raw/cosmx/raw/Lung5*/*/*_Z001.TIF path/to/data/raw/cosmx/train/
ln -s path/to/data/raw/cosmx/raw/Lung9*/*/*_Z001.TIF path/to/data/raw/cosmx/train/
ln -s path/to/data/raw/cosmx/raw/Lung12*/*/*_Z001.TIF path/to/data/raw/cosmx/train/
```

Now we can execute Image2Count:  
```sh
./reproduce/cosmx/cosmx.sh
```

To get (more accurate?) single cell correlation we impute false zero counts with the dca method:  
```sh
conda create -n dca
conda activate dca
conda install keras=2.4.3 tensorflow=2.4.0 scanpy dca
python src/utils/impute_w_dca.py
mkdir data/raw/cosmx_dca/
mkdir data/raw/cosmx/test/
mkdir data/raw/cosmx/train/
ln -s path/to/data/raw/cosmx/*npy data/raw/cosmx_dca/
ln -s path/to/data/raw/cosmx/train/* data/raw/cosmx_dca/train/
ln -s path/to/data/raw/cosmx/test/* data/raw/cosmx_dca/test/
ln -s path/to/data/raw/cosmx_measuremens_flipped_y_dca.csv data/raw/cosmx_dca/
ln -s path/to/data/raw/cosmx_measuremens_flipped_y_dca.csv data/raw/
./reproduce/cosmx/dca.sh
```
Please know that getting dca to work took some manual work adjusting version numbers of packages, you might need to do the sam eif you run into errors.  

This file creates `.csv` files for each model for all number of neighbourhood hops (hop is identified as the number before `[].csv`, otherwise it is a single cell metric file). Model name is in the file name. Metrics include Pearson, Spearman and Kendall correlation, Mutual Information, Cosinse Similarity and Mean squared error of log1p expression values and more. 