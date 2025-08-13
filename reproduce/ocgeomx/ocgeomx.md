# OC GeoMx

Follow the instructions to intall required packages and become familiar with Image2Count.  

We have Image data of 636 ROIs with corresponding GeoMx count data previously normalized through geometric mean of H3, cell postions, and the Image data of the 1C54 TMA. Installing QuPath by following the [instructions](https://qupath.readthedocs.io/en/stable/docs/reference/building.html) to enable GPU usage, and install the Stardist extension following the [instructions](https://qupath.readthedocs.io/en/stable/docs/deep/stardist.html), loading extensions is done via letting QuPath execute a modified version of the script in `src/utils/qupath_include_extension.groovy` that points to a directory which has a subdir called extension, this subdir contains extension `.jar`. Segmentation is done with the script `src/utils/qupath_segment.groovy`, modify path to stardist model and channel name of `.tiff` files to segment. Script can be executed as follows after creating ROIs:

```sh
./qupath/build/dist/QuPath/bin/QuPath script -p=1C-54/project.qpproj segment.groovy
```

With `1C-54/` being the directory containing the QuPath project.  
ROI image data is exported through executing the script `src/utils/qupath_create_roi_tiffs.groovy`. Cell postions get exported manualy and transformed to the correct format. Cell count information and cell segmentationdata of the 636 GeoMx ROIs is manualy transformed into the correct format, also utilizing script `src/utils/getRelevantCSVData.py`. For visual representation learning we combine label and measurements `.csv` of ROIs and 1C54 and create a directory holding the combined tiff files.  
We split data into train and test:
```sh
mv data/raw/hkgmh3_74/*1B65* data/raw/hkgmh3_74/test/
mv data/raw/hkgmh3_74/*1C54* data/raw/hkgmh3_74/test/
mv data/raw/hkgmh3_74/*3B18* data/raw/hkgmh3_74/test/
mv data/raw/hkgmh3_74/*.tiff data/raw/hkgmh3_74/train/
```

We then learn from the provided data:

```sh
./reproduce/ocgeomx/hkgmh3_74_20.sh
./reproduce/ocgeomx/hkgmh3_74_30.sh
./reproduce/ocgeomx/hkgmh3_74_50.sh
```

This only produces predictions for provided GeoMx ROIs. To create predictions for other images create new experiment folder containing images you want to predict for in `test`, create corresponding `measurements.csv` and `labels.csv`. For our case we added segmented cores to test GeoMx ROIs and renamed GeoMX ROIs file names and entries in `.csv` to `X-{NAME}`, then visualize corresponding expression.

To reproduce single cell metrics you will need to modify `src/utils/create_metrics_from_h5ad` as follows:  
```py
path = 'path/to/crossvalidation/model/h5ad/saves/'  # Path to dir containing .h5ad files of predicted data
target = 'csv/containing/true/expression/data.csv'
num_subgraphs_per_graph = 900   # As named
num_hops_per_subgraph = []  # List of hops to expand cellular neighbourhood, e.g. [1, 2, 3, 5, 8, 11]

name = 'regular_expression_pattern_of_crossval_model_h5ad_files.h5ad'   # As assigned
out = 'figures/path/to/save/'   # As assigned

sum_by_graph = False    # If you use a path pointing to measurements.csv of an experiemnt, meaning measurements.csv
                        # contains per cell gene expression, set this to False. If path points to labels.csv, containing
                        # expression per image, set this to True.
```  
This should be turned to:  
```py
path = 'out/'
target = 'data/raw/hkgmh3_74/label.csv'
num_subgraphs_per_graph = 0 
num_hops_per_subgraph = []  

name = 'hkgmh3_74_20_3_3_[0-9]_all.h5ad'   #hkgmh3_74_20_3_3_mean_all.h5ad for the merged prediction data
out = 'figures/hkgmh3_74_20/3_3/'

sum_by_graph = True

``` 

This file creates `.csv` files for each model. Model name is in the file name. Metrics include Pearson, Spearman and Kendall correlation, Mutual Information, Cosinse Similarity and Mean squared error of log1p expression values. Needs to be executed once to create metrics for each model, and once to create metrics for the merged model predictions. Every time you execute `src/utils/create_metrics_from_h5ad` you have to save `.csv` files in a new directory, otherwise calculated metrics in the next step might be wrong.  
To give an average model performance over all crossvalidation runs run `src/utils/mean_subgraph_metrics.py` by changing code as follows:
```py
path = 'figures/hkgmh3_74_20/3_3/'
subgraphs = []
out = 'figures/hkgmh3_74_20/3_3/'
```  
This will create `.csv` files for each neighbourhood hop size with avage performance metrics for all crossvalidation model runs.  

`src/utils/create_metrics_from_h5ad` and `src/utils/mean_subgraph_metrics.py` need to be executed by model type ( so 3*2 times: for the gat, ffw and lin model and their mean predictions by changing the `name` varible appropriatly, as well as where figures are saved). This needs to be done per experiment (3 times total).