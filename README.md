# ASIFT++:
Descriptor based on SIFT which is robust to *out-of-plane* rotations. ASIFT properly deals only with in-plane rotations. I used OpenCV implementation of SIFT. How does it work:
- find keypoints in 1st frame using specified detector
- describe them using specified descriptor
- create vector of descriptors for each keypoint using perspective transformations (that is why we need chessboard files under variety of rotations).
- detect and describe keypoints on 2nd frame
- match 2nd frame keypoints with _vectors_ od descriptors for keypoints from 1st frame

# Focal adjustment [19.07.2017]:
I presented novelty in adapted 3D reconstruction alghoritm (EPFL). There are alghoritms that optimize extrinsic camera parameters, but - in the knowledge of authors - noone tried to estimate the intrinsic parameters. In unconstrained reconstruction step I track displacement of reprojected points in each iteration. If the displacement in two successive iterations shows that the point is closer to the middle of the image - I assume that focal length could be a little bit too long (somehow the distance from camera is expressed as a function of focal length). For better reprojection in next iteration I adjust the focal length and make it smaller. Otherwise, this parameter is enlarged. 

# Robust to in-plane rotation alghoritm (ASIFT):
http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html

# Adapted 3D reconstruction alghoritm:
http://cvlab.epfl.ch/files/content/sites/cvlab2/files/publications/publications/2012/OstlundVF12.pdf

# BUILD:
- build using standard procedure (mkdir build -> cd build -> cmake .. -> make)
- in _build_ folder prepare folder _images_ and create here _chessboard.txt_ wher you list all your chessboard files (needed for initial deformation) - examplary images are in _images_ folder from this repo.
- Paste to _build_ folder all files from dataset (EPFL): cam.ext, cam.intr, cam.tdir, ControlPointIds.txt, im_corners.txt, mesh.pts, mesh.tri, webcam.intr, world_corners.txt
- Usage: ./affineDSC path_to_model.png path_to_frame.png detector descriptor ratio1% ratio2% isPointCloudSaved(0 / 1)

# LOG:
- [19.06.2017] Focal adjustment added + few minor improvements.

# RESULTS USING SIFT:
- All matches found:
![All matches found](https://raw.githubusercontent.com/mbed92/ASIFTplusplus/master/PC_056_80_90_sift_sift_all.png)

- Displacement of added keypoints:
![Displacement of improved keypoints](https://raw.githubusercontent.com/mbed92/ASIFTplusplus/master/PC_056_80_90_sift_sift_disp.png)

- Position of added keypoints:
![Added matches](https://raw.githubusercontent.com/mbed92/ASIFTplusplus/master/PC_056_80_90_sift_sift_imp.png)
