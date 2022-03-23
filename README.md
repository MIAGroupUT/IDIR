# IDIR
Code for the MIDL 2022 paper [Implicit Neural Representations for Deformable Image Registration](https://openreview.net/forum?id=BP29eKzQBu3). In this work, we register medical images using differentiable deformation vector fields represented in multilayer perceptrons. We show how this allows us to include various regularization terms computed using analytical gradients in PyTorch.

![Method overview!](Overview.png "Method overview")

# Running the code
This code replicates the experiments that we ran on the DIR-LAB data set. You will need [PyTorch](https://pytorch.org/) to run the code. By default, the code expects a CUDA-enabled GPU.  To register inspiration and expiration images for a patient in this set, just run `run.py`. You can turn the different regularizers on and off by modifying the script, and choose to use either a SIREN or MLP (faster) network. More advanced settings can be changed in `models\models.py`. As output, you will get the mean and standard deviation of the target registration (TRE) error for the 300 anatomical landmarks in Euclidean distance as well as per axis. 

# Data
We have used data from the 4D CT DIR-LAB set in our experiments. You can obtain this data from the [DIR-LAB website](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html). Note that our script expects filenames to have a standardized naming convention: we assume that for each patient there is an image `case{}_T00_s.img` and an image `case{}_T50_s.img` for inspiration and expiration, respectively. Moreover, we use lung masks that we obtain using the excellent scripts provided by Johannes Hofmanninger on his [GitHub page](https://github.com/JoHof/lungmask). You can of course also use different lung masks or different file formats. As long as you adhere to the file structure below, things should run smoothly. You should set `data_dir` in `run.py`.

📦data_dir  
 ┣ 📂Case1Pack  
 ┃ ┣ 📂ExtremePhases  
 ┃ ┃ ┣ 📜Case1_300_T00_xyz.txt  
 ┃ ┃ ┗ 📜Case1_300_T50_xyz.txt  
 ┃ ┣ 📂Images  
 ┃ ┃ ┣ 📜case1_T00_s.img  
 ┃ ┃ ┗ 📜case1_T50_s.img  
 ┃ ┗ 📂Masks  
 ┃ ┃ ┣ 📜case1_T00_s.mhd  
 ┃ ┃ ┗ 📜case1_T00_s.raw  
 ┣ 📂Case2Pack  
 ┃ ┣..  
 ┃..  
 ┣ 📂Case10Pack  
 ┃ ┣ 📂extremePhases  
 ┃ ┃ ┣ 📜Case10_300_T00_xyz.txt  
 ┃ ┃ ┗ 📜Case10_300_T50_xyz.txt  
 ┃ ┣ 📂Images  
 ┃ ┃ ┣ 📜case10_T00_s.img  
 ┃ ┃ ┗ 📜case10_T50_s.img  
 ┃ ┗ 📂Masks  
 ┃ ┃ ┣ 📜case10_T00_s.mhd  
 ┃ ┃ ┗ 📜case10_T00_s.raw  

# Reference
If you use this code, please cite our MIDL 2022 paper

    @inproceedings{wolterink2021implicit,
      title={Implicit Neural Representations for Deformable Image Registration},
      author={Wolterink, Jelmer M and Zwienenberg, Jesse C and Brune, Christoph},
      booktitle={Medical Imaging with Deep Learning 2022}
      year={2022}
    }
