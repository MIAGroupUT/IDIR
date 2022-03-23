# IDIR
Code for the MIDL 2022 paper [Implicit Neural Representations for Deformable Image Registration](https://openreview.net/forum?id=BP29eKzQBu3). In this work, we register medical images using differentiable deformation vector fields represented in multilayer perceptrons. We show how this allows us to include various regularization terms computed using analytical gradients in PyTorch.

![Method overview!](Overview.png "Method overview")

# Data
We have used data from the 4D CT DIR-LAB set in our experiments. You can obtain this data from the [DIR-LAB website](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html). Note that our script expects filenames to have a standardized naming convention: we assume that for each patient there is an image `case{}_T00_s.img` and an image `case{}_T50_s.img` for inspiration and expiration, respectively. 

 📦DIRLAB
  ┣ 📂Case10Pack
  ┃ ┣ 📂extremePhases
  ┃ ┃ ┣ 📜Case10_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case10_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case10_T00_s.img
  ┃ ┃ ┣ 📜case10_T00_s.mhd
  ┃ ┃ ┣ 📜case10_T00_s.zraw
  ┃ ┃ ┣ 📜case10_T10.img
  ┃ ┃ ┣ 📜case10_T20.img
  ┃ ┃ ┣ 📜case10_T30.img
  ┃ ┃ ┣ 📜case10_T40.img
  ┃ ┃ ┣ 📜case10_T50_s.img
  ┃ ┃ ┣ 📜case10_T50_s.mhd
  ┃ ┃ ┣ 📜case10_T50_s.zraw
  ┃ ┃ ┣ 📜case10_T60.img
  ┃ ┃ ┣ 📜case10_T70.img
  ┃ ┃ ┣ 📜case10_T80.img
  ┃ ┃ ┗ 📜case10_T90.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case10_T00_s.mhd
  ┃ ┃ ┣ 📜case10_T00_s.raw
  ┃ ┃ ┣ 📜case10_T50_s.mhd
  ┃ ┃ ┗ 📜case10_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case10_4D-75_T00.txt
  ┃ ┃ ┣ 📜case10_4D-75_T10.txt
  ┃ ┃ ┣ 📜case10_4D-75_T20.txt
  ┃ ┃ ┣ 📜case10_4D-75_T30.txt
  ┃ ┃ ┣ 📜case10_4D-75_T40.txt
  ┃ ┃ ┗ 📜case10_4D-75_T50.txt
  ┣ 📂Case1Pack
  ┃ ┣ 📂ExtremePhases
  ┃ ┃ ┣ 📜Case1_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case1_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case1_T00_s.img
  ┃ ┃ ┣ 📜case1_T00_s.mhd
  ┃ ┃ ┣ 📜case1_T00_s.zraw
  ┃ ┃ ┣ 📜case1_T10_s.img
  ┃ ┃ ┣ 📜case1_T20_s.img
  ┃ ┃ ┣ 📜case1_T30_s.img
  ┃ ┃ ┣ 📜case1_T40_s.img
  ┃ ┃ ┣ 📜case1_T50_s.img
  ┃ ┃ ┣ 📜case1_T50_s.mhd
  ┃ ┃ ┣ 📜case1_T50_s.zraw
  ┃ ┃ ┣ 📜case1_T60_s.img
  ┃ ┃ ┣ 📜case1_T70_s.img
  ┃ ┃ ┣ 📜case1_T80_s.img
  ┃ ┃ ┗ 📜case1_T90_s.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case1_T00_s.mhd
  ┃ ┃ ┣ 📜case1_T00_s.raw
  ┃ ┃ ┣ 📜case1_T50_s.mhd
  ┃ ┃ ┗ 📜case1_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case1_4D-75_T00.txt
  ┃ ┃ ┣ 📜case1_4D-75_T10.txt
  ┃ ┃ ┣ 📜case1_4D-75_T20.txt
  ┃ ┃ ┣ 📜case1_4D-75_T30.txt
  ┃ ┃ ┣ 📜case1_4D-75_T40.txt
  ┃ ┃ ┗ 📜case1_4D-75_T50.txt
  ┣ 📂Case2Pack
  ┃ ┣ 📂ExtremePhases
  ┃ ┃ ┣ 📜Case2_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case2_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case2_T00_s.img
  ┃ ┃ ┣ 📜case2_T00_s.mhd
  ┃ ┃ ┣ 📜case2_T00_s.zraw
  ┃ ┃ ┣ 📜case2_T10-s.img
  ┃ ┃ ┣ 📜case2_T20-s.img
  ┃ ┃ ┣ 📜case2_T30-s.img
  ┃ ┃ ┣ 📜case2_T40-s.img
  ┃ ┃ ┣ 📜case2_T50_s.img
  ┃ ┃ ┣ 📜case2_T50_s.mhd
  ┃ ┃ ┣ 📜case2_T50_s.zraw
  ┃ ┃ ┣ 📜case2_T60-s.img
  ┃ ┃ ┣ 📜case2_T70-s.img
  ┃ ┃ ┣ 📜case2_T80-s.img
  ┃ ┃ ┗ 📜case2_T90-s.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case2_T00_s.mhd
  ┃ ┃ ┣ 📜case2_T00_s.raw
  ┃ ┃ ┣ 📜case2_T50_s.mhd
  ┃ ┃ ┗ 📜case2_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case2_4D-75_T00.txt
  ┃ ┃ ┣ 📜case2_4D-75_T10.txt
  ┃ ┃ ┣ 📜case2_4D-75_T20.txt
  ┃ ┃ ┣ 📜case2_4D-75_T30.txt
  ┃ ┃ ┣ 📜case2_4D-75_T40.txt
  ┃ ┃ ┗ 📜case2_4D-75_T50.txt
  ┣ 📂Case3Pack
  ┃ ┣ 📂ExtremePhases
  ┃ ┃ ┣ 📜Case3_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case3_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case3_T00_s.img
  ┃ ┃ ┣ 📜case3_T00_s.mhd
  ┃ ┃ ┣ 📜case3_T00_s.zraw
  ┃ ┃ ┣ 📜case3_T10-ssm.img
  ┃ ┃ ┣ 📜case3_T20-ssm.img
  ┃ ┃ ┣ 📜case3_T30-ssm.img
  ┃ ┃ ┣ 📜case3_T40-ssm.img
  ┃ ┃ ┣ 📜case3_T50_s.img
  ┃ ┃ ┣ 📜case3_T50_s.mhd
  ┃ ┃ ┣ 📜case3_T50_s.zraw
  ┃ ┃ ┣ 📜case3_T60-ssm.img
  ┃ ┃ ┣ 📜case3_T70-ssm.img
  ┃ ┃ ┣ 📜case3_T80-ssm.img
  ┃ ┃ ┗ 📜case3_T90-ssm.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case3_T00_s.mhd
  ┃ ┃ ┣ 📜case3_T00_s.raw
  ┃ ┃ ┣ 📜case3_T50_s.mhd
  ┃ ┃ ┗ 📜case3_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case3_4D-75_T00.txt
  ┃ ┃ ┣ 📜case3_4D-75_T10.txt
  ┃ ┃ ┣ 📜case3_4D-75_T20.txt
  ┃ ┃ ┣ 📜case3_4D-75_T30.txt
  ┃ ┃ ┣ 📜case3_4D-75_T40.txt
  ┃ ┃ ┗ 📜case3_4D-75_T50.txt
  ┣ 📂Case4Pack
  ┃ ┣ 📂ExtremePhases
  ┃ ┃ ┣ 📜Case4_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case4_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case4_T00_s.img
  ┃ ┃ ┣ 📜case4_T00_s.mhd
  ┃ ┃ ┣ 📜case4_T00_s.zraw
  ┃ ┃ ┣ 📜case4_T10-ssm.img
  ┃ ┃ ┣ 📜case4_T20-ssm.img
  ┃ ┃ ┣ 📜case4_T30-ssm.img
  ┃ ┃ ┣ 📜case4_T40-ssm.img
  ┃ ┃ ┣ 📜case4_T50_s.img
  ┃ ┃ ┣ 📜case4_T50_s.mhd
  ┃ ┃ ┣ 📜case4_T50_s.zraw
  ┃ ┃ ┣ 📜case4_T60-ssm.img
  ┃ ┃ ┣ 📜case4_T70-ssm.img
  ┃ ┃ ┣ 📜case4_T80-ssm.img
  ┃ ┃ ┗ 📜case4_T90-ssm.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case4_T00_s.mhd
  ┃ ┃ ┣ 📜case4_T00_s.raw
  ┃ ┃ ┣ 📜case4_T50_s.mhd
  ┃ ┃ ┗ 📜case4_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case4_4D-75_T00.txt
  ┃ ┃ ┣ 📜case4_4D-75_T10.txt
  ┃ ┃ ┣ 📜case4_4D-75_T20.txt
  ┃ ┃ ┣ 📜case4_4D-75_T30.txt
  ┃ ┃ ┣ 📜case4_4D-75_T40.txt
  ┃ ┃ ┗ 📜case4_4D-75_T50.txt
  ┣ 📂Case5Pack
  ┃ ┣ 📂ExtremePhases
  ┃ ┃ ┣ 📜Case5_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case5_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case5_T00_s.img
  ┃ ┃ ┣ 📜case5_T00_s.mhd
  ┃ ┃ ┣ 📜case5_T00_s.zraw
  ┃ ┃ ┣ 📜case5_T10-ssm.img
  ┃ ┃ ┣ 📜case5_T20-ssm.img
  ┃ ┃ ┣ 📜case5_T30-ssm.img
  ┃ ┃ ┣ 📜case5_T40-ssm.img
  ┃ ┃ ┣ 📜case5_T50_s.img
  ┃ ┃ ┣ 📜case5_T50_s.mhd
  ┃ ┃ ┣ 📜case5_T50_s.zraw
  ┃ ┃ ┣ 📜case5_T60-ssm.img
  ┃ ┃ ┣ 📜case5_T70-ssm.img
  ┃ ┃ ┣ 📜case5_T80-ssm.img
  ┃ ┃ ┗ 📜case5_T90-ssm.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case5_T00_s.mhd
  ┃ ┃ ┣ 📜case5_T00_s.raw
  ┃ ┃ ┣ 📜case5_T50_s.mhd
  ┃ ┃ ┗ 📜case5_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case5_4D-75_T00.txt
  ┃ ┃ ┣ 📜case5_4D-75_T10.txt
  ┃ ┃ ┣ 📜case5_4D-75_T20.txt
  ┃ ┃ ┣ 📜case5_4D-75_T30.txt
  ┃ ┃ ┣ 📜case5_4D-75_T40.txt
  ┃ ┃ ┗ 📜case5_4D-75_T50.txt
  ┣ 📂Case6Pack
  ┃ ┣ 📂extremePhases
  ┃ ┃ ┣ 📜Case6_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case6_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case6_T00_s.img
  ┃ ┃ ┣ 📜case6_T00_s.mhd
  ┃ ┃ ┣ 📜case6_T00_s.zraw
  ┃ ┃ ┣ 📜case6_T10.img
  ┃ ┃ ┣ 📜case6_T20.img
  ┃ ┃ ┣ 📜case6_T30.img
  ┃ ┃ ┣ 📜case6_T40.img
  ┃ ┃ ┣ 📜case6_T50_s.img
  ┃ ┃ ┣ 📜case6_T50_s.mhd
  ┃ ┃ ┣ 📜case6_T50_s.zraw
  ┃ ┃ ┣ 📜case6_T60.img
  ┃ ┃ ┣ 📜case6_T70.img
  ┃ ┃ ┣ 📜case6_T80.img
  ┃ ┃ ┗ 📜case6_T90.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case6_T00_s.mhd
  ┃ ┃ ┣ 📜case6_T00_s.raw
  ┃ ┃ ┣ 📜case6_T50_s.mhd
  ┃ ┃ ┗ 📜case6_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case6_4D-75_T00.txt
  ┃ ┃ ┣ 📜case6_4D-75_T10.txt
  ┃ ┃ ┣ 📜case6_4D-75_T20.txt
  ┃ ┃ ┣ 📜case6_4D-75_T30.txt
  ┃ ┃ ┣ 📜case6_4D-75_T40.txt
  ┃ ┃ ┗ 📜case6_4D-75_T50.txt
  ┣ 📂Case7Pack
  ┃ ┣ 📂extremePhases
  ┃ ┃ ┣ 📜Case7_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case7_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case7_T00_s.img
  ┃ ┃ ┣ 📜case7_T00_s.mhd
  ┃ ┃ ┣ 📜case7_T00_s.zraw
  ┃ ┃ ┣ 📜case7_T10.img
  ┃ ┃ ┣ 📜case7_T20.img
  ┃ ┃ ┣ 📜case7_T30.img
  ┃ ┃ ┣ 📜case7_T40.img
  ┃ ┃ ┣ 📜case7_T50_s.img
  ┃ ┃ ┣ 📜case7_T50_s.mhd
  ┃ ┃ ┣ 📜case7_T50_s.zraw
  ┃ ┃ ┣ 📜case7_T60.img
  ┃ ┃ ┣ 📜case7_T70.img
  ┃ ┃ ┣ 📜case7_T80.img
  ┃ ┃ ┗ 📜case7_T90.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case7_T00_s.mhd
  ┃ ┃ ┣ 📜case7_T00_s.raw
  ┃ ┃ ┣ 📜case7_T50_s.mhd
  ┃ ┃ ┗ 📜case7_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case7_4D-75_T00.txt
  ┃ ┃ ┣ 📜case7_4D-75_T10.txt
  ┃ ┃ ┣ 📜case7_4D-75_T20.txt
  ┃ ┃ ┣ 📜case7_4D-75_T30.txt
  ┃ ┃ ┣ 📜case7_4D-75_T40.txt
  ┃ ┃ ┗ 📜case7_4D-75_T50.txt
  ┣ 📂Case8Pack
  ┃ ┣ 📂extremePhases
  ┃ ┃ ┣ 📜Case8_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case8_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case8_T00_s.img
  ┃ ┃ ┣ 📜case8_T00_s.mhd
  ┃ ┃ ┣ 📜case8_T00_s.zraw
  ┃ ┃ ┣ 📜case8_T10.img
  ┃ ┃ ┣ 📜case8_T20.img
  ┃ ┃ ┣ 📜case8_T30.img
  ┃ ┃ ┣ 📜case8_T40.img
  ┃ ┃ ┣ 📜case8_T50_s.img
  ┃ ┃ ┣ 📜case8_T50_s.mhd
  ┃ ┃ ┣ 📜case8_T50_s.zraw
  ┃ ┃ ┣ 📜case8_T60.img
  ┃ ┃ ┣ 📜case8_T70.img
  ┃ ┃ ┣ 📜case8_T80.img
  ┃ ┃ ┗ 📜case8_T90.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case8_T00_s.mhd
  ┃ ┃ ┣ 📜case8_T00_s.raw
  ┃ ┃ ┣ 📜case8_T50_s.mhd
  ┃ ┃ ┣ 📜case8_T50_s.raw
  ┃ ┃ ┗ 📜mask0.stl
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case8_4D-75_T00.txt
  ┃ ┃ ┣ 📜case8_4D-75_T10.txt
  ┃ ┃ ┣ 📜case8_4D-75_T20.txt
  ┃ ┃ ┣ 📜case8_4D-75_T30.txt
  ┃ ┃ ┣ 📜case8_4D-75_T40.txt
  ┃ ┃ ┗ 📜case8_4D-75_T50.txt
  ┗ 📂Case9Pack
  ┃ ┣ 📂extremePhases
  ┃ ┃ ┣ 📜Case9_300_T00_xyz.txt
  ┃ ┃ ┗ 📜Case9_300_T50_xyz.txt
  ┃ ┣ 📂Images
  ┃ ┃ ┣ 📜case9_T00_s.img
  ┃ ┃ ┣ 📜case9_T00_s.mhd
  ┃ ┃ ┣ 📜case9_T00_s.zraw
  ┃ ┃ ┣ 📜case9_T10.img
  ┃ ┃ ┣ 📜case9_T20.img
  ┃ ┃ ┣ 📜case9_T30.img
  ┃ ┃ ┣ 📜case9_T40.img
  ┃ ┃ ┣ 📜case9_T50_s.img
  ┃ ┃ ┣ 📜case9_T50_s.mhd
  ┃ ┃ ┣ 📜case9_T50_s.zraw
  ┃ ┃ ┣ 📜case9_T60.img
  ┃ ┃ ┣ 📜case9_T70.img
  ┃ ┃ ┣ 📜case9_T80.img
  ┃ ┃ ┗ 📜case9_T90.img
  ┃ ┣ 📂Masks
  ┃ ┃ ┣ 📜case9_T00_s.mhd
  ┃ ┃ ┣ 📜case9_T00_s.raw
  ┃ ┃ ┣ 📜case9_T50_s.mhd
  ┃ ┃ ┗ 📜case9_T50_s.raw
  ┃ ┗ 📂Sampled4D
  ┃ ┃ ┣ 📜case9_4D-75_T00.txt
  ┃ ┃ ┣ 📜case9_4D-75_T10.txt
  ┃ ┃ ┣ 📜case9_4D-75_T20.txt
  ┃ ┃ ┣ 📜case9_4D-75_T30.txt
  ┃ ┃ ┣ 📜case9_4D-75_T40.txt
  ┃ ┃ ┗ 📜case9_4D-75_T50.txt

# Reference
If you use this code, please cite our MIDL 2022 paper

    @inproceedings{wolterink2021implicit,
      title={Implicit Neural Representations for Deformable Image Registration},
      author={Wolterink, Jelmer M and Zwienenberg, Jesse C and Brune, Christoph},
      booktitle={Medical Imaging with Deep Learning 2022}
      year={2022}
    }


