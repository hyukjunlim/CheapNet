Training a GNN model for LEP data
==================================


Installation
------------

The GNN models require Pytorch Geometric — see details in either `ATOM3D <https://atom3d.readthedocs.io/en/latest/training_models.html#model-specific-installation-instructions>`_ or `PTG <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ documentation.

Dataset
-------


Download the *split* LEP datasets from `the ATOM3D website <https://www.atom3d.ai/lep.html>`_.
Once the download has finished, extract the datasets from the zipped archive.


Training
--------
  
The training script can be invoked from the example folder using, e.g.::

    cd atom3d/examples/lep/gnn
    python train.py --learning_rate 15e-4 --seed_set 1 --GPU_NUM 0
                    
where LMDBDIR is the path to the subfolder "/data" of the split LMDB dataset.

To see further options for training, use::

    python train.py --help
