# Key novelties
## [23.05.2017] Simulated local deformation:
Descriptor robust to *out-of-plane* rotations by performing similar approach as in ASIFT article - simulated *local* deformation. I used OpenCV implementation of SIFT. How does it work:
- find keypoints and descriptors on reference frame
- simulated local deformation - add a stack of descriptors for each keypoint using perspective transformations (that is why we need chessboard files under variety of rotations).
- detect and describe keypoints on test frame
- perform matching descriptors from test frame with *stac* of descriptors from reference image - improved matching process
- profit

## [19.06.2017] Focal adjustment:
I presented novelty in adapted 3D reconstruction alghoritm (EPFL). There are alghoritms that optimize extrinsic camera parameters, but - in the knowledge of authors - noone tried to estimate the intrinsic parameters. In unconstrained reconstruction step I track displacement of reprojected points in each iteration. If the displacement in two successive iterations shows that the point is closer to the middle of the image - I assume that focal length is a little bit too long (somehow the distance from camera is expressed as a function of focal length). For better reprojection in next iteration I adjust the focal length and make it smaller. Otherwise, this parameter is enlarged. 

# Used ideas
## Robust to in-plane rotation alghoritm (ASIFT):
http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html

## Adapted 3D reconstruction alghoritm:
http://cvlab.epfl.ch/files/content/sites/cvlab2/files/publications/publications/2012/OstlundVF12.pdf

# HOWTO
## Build:
- build using standard procedure (mkdir build -> cd build -> cmake .. -> make)
- in _build_ create *chessboard.txt* file where are listed all images needed for simulated local deformation. If the size of chessboard patter is other than 9x6, change it in _void KpMatcher::findCorners_.
- Paste to _build_ folder all files from dataset (EPFL): cam.ext, cam.intr, cam.tdir, ControlPointIds.txt, im_corners.txt, mesh.pts, mesh.tri, webcam.intr, world_corners.txt
- Usage: ./affineDSC path_to_model.png path_to_frame.png detector descriptor ratio1% ratio2% isPointCloudSaved(0 / 1)

# Results - simulated local deformation
## SIFT was used for evaluation purposes:
- All matches found:
![All matches found](https://raw.githubusercontent.com/mbed92/ASIFTplusplus/master/PC_056_80_90_sift_sift_all.png)

- Displacement of added keypoints:
![Displacement of improved keypoints](https://raw.githubusercontent.com/mbed92/ASIFTplusplus/master/PC_056_80_90_sift_sift_disp.png)

- Position of added keypoints:
![Added matches](https://raw.githubusercontent.com/mbed92/ASIFTplusplus/master/PC_056_80_90_sift_sift_imp.png)

# Milestones:
- [09.03.17] Adapted EPFL code for my own purposes. 
- [23.05.17] Simulated local deformation - stable version.
- [19.06.17] Focal adjustment added + few minor improvements.
