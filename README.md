# ASIFT++
Trying to improve ASIFT alghoritm.

# Original paper
http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html

# EPFL
Reconstruction alghoritm:
http://cvlab.epfl.ch/files/content/sites/cvlab2/files/publications/publications/2012/OstlundVF12.pdf

Now I'm testing and trying to understand how to do that.. ;) 3D reconstruction is needed for evaluation purposes (to check if additional matches improved reconstruction).

# BUILD
- build using standard procedure (mkdir build -> cd build -> cmake .. -> make)
- in _build_ folder prepare folder _images_ and create here _chessboard.txt_ wher you list all your chessboard files (needed for initial deformation).
- Paste to _build_ folder all files from dataset (EPFL): cam.ext, cam.intr, cam.tdir, ControlPointIds.txt, im_corners.txt, mesh.pts, mesh.tri, webcam.intr, world_corners.txt
- Usage: ./affineDSC path_to_model.png path_to_frame.png detector descriptor ratio1% ratio2% isPointCloudSaved(0 / 1)
