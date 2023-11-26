File: README.txt
Dataset: SegSperm
License: CC BY-NC 4.0
------------------------------------
The SegSperm dataset contains microscopic 
gray images of sperm with the associated 
binary segmentation masks. The segmentation
masks were created manually with the help of
GIMP software (www.gimp.org). The images 
were collected by Invicta (www.invictaclinics.com).

The dataset contains the following folders:
1. [test]
2. [test_small]
3. [train]

The [test] folder contains 119 sperm images 
and 119 segmentation masks (annotator GT1).

The [test_small] folder contains 23 sperm images 
and 23 segmentation masks of head, tail and full sperm, 
manually prepared by 3 raters (annotators GT1, GT2, GT3).

The [train] folder contains 432 sperm images 
and 432 segmentation masks (annotator GT1).

If you use the SegSperm dataset in your research, 
please cite the following paper:
-------------------------------
@article{lewandowska2023ensembling,
  title={Ensembling noisy segmentation masks of blurred sperm images},
  author={Lewandowska, Emilia and W{\k{e}}sierski, Daniel and Mazur-Milecka, 
		  Magdalena and Liss, Joanna and Jezierska, Anna},
  journal={Computers in Biology and Medicine},
  pages={107520},
  year={2023},
  publisher={Elsevier}
}
doi: https://doi.org/10.1016/j.compbiomed.2023.107520
-------------------------------