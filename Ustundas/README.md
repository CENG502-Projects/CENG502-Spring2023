# SPARF: Neural Radiance Fields from Sparse and Noisy Poses

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).
SPARF: Neural Radiance Fields from Sparse and Noisy Poses is a NeRF-family paper published in CVPR2023, by Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt and Federico Tombari.

In this paper, the authors inspect the effect of the pose quality and quantity on the overall quality of a Neural Radiance Field. They state that unavailibity of dense input views and inaccurate camera pose data limit the real-world application of NeRFs. To address these challenges, they propose SPARF, a Novel-view Synthesis model given only few input images, and also noisy camera poses.

In this project, I will try to implement the Joint Pose-NeRF training method described in the paper, and reproduce their results.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

In order to run the training procedure, use the following snippet:
```
	python train.py <scene>
```
Where <scene> is a scan from a preprocessed version of DTU dataset, taken from PixelNeRF.[pixelNerf][2]
@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

[1] Truong, P, Rakotosaona, M, Manhardt, F, Tombari, F. "SPARF
Neural Radiance Fields from Sparse and Noisy Poses". (CVPR) 2023

[2] Yu, A, Ye, V, Tancik, M, Kanazawa, A. "pixelNeRF
Neural Radiance Fields from One or Few Images". (CVPR) 2021

[3] Lin, C, Ma, W, Torralba, A, Lucey, S. "BARF: Bundle-Adjusting Neural Radiance Fields". (ICCV) 2021

[4] Mildenhall, B, Srinivasan, P, Tancik, M, Barron, J, Ramamoorthi, R, Ng, R. "NeRF
Representing Scenes as Neural Radiance Fields for View Synthesis". (ECCV) 2020


# Contact

Şahin Umutcan Üstündaş (umutcan.ustundas@metu.edu.tr)