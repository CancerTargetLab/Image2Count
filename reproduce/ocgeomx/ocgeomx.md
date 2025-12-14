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

Reproduction creates `.csv` files for each model. Model name is in the file name. Metrics include Pearson, Spearman and Kendall correlation, Mutual Information, Cosinse Similarity and Mean squared error of log1p expression values and more.  
