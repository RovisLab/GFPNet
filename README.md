# GFPNet
## A Single-view 3D Object Reconstruction using Generic Fitted Primitives
3D volumetric reconstruction from incomplete
point clouds remains one of the fundamental challenges in
perception. One solution is to register an a-priori defined
shape (i.e. a CAD model) onto the incomplete point cloud of
the object. The subsequent challenge that arises is to identify
the most similar shape of the object from a database of
shapes. In this paper, we propose a volumetric reconstruction
apparatus that uses so-called Generic Primitive Shapes (GFP)
to abstract the large variety of shapes that an object may
have. We use a kernel-based deformation technique to fit a
GFP to real-world objects, where the kernel is encoded within
the layers of a Deep Neural Network (DNN). The objective
is to transfer the particularities of the perceived object to
the raw volume represented by the generic primitive. We
show that GPFNet outperforms competing algorithms on 3D
volume reconstruction challenges applied on the ModelNet and
KITTI benchmarking datasets. GFPNet is compared with a
baseline approach, as well as with state-of-the-art data-driven
approaches for volumetric reconstruction