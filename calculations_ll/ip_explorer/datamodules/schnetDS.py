from .base import PLDataModuleWrapper

import os
import ast
import numpy as np

import schnetpack.transform as tform
from schnetpack.data import AtomsLoader, AtomsDataFormat
from schnetpack.data.datamodule import AtomsDataModule


class SchNetDSDataModule(PLDataModuleWrapper):
    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                Path to a directory containing the following files:

                    * `<database_name>.db`: the formatted schnetpack.data.ASEAtomsData database
                    * `split.npz`: the file specifying the train/val/test split indices
        """
        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for SchNetDataModule. Use --additional-kwargs argument.")
            
        if 'database_name' not in kwargs:
            raise RuntimeError("Must specify database_name for SchNetDataModule. Use --additional-datamodule-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])
        self.database_name = str(kwargs['database_name'])

        if 'remove_offsets' not in kwargs:
            self.remove_offsets = True
        else:
            self.remove_offsets = ast.literal_eval(kwargs['remove_offsets'])

        if 'train_filename' in kwargs:
            self.train_filename = kwargs['train_filename']
        else:
            self.train_filename = None
        if 'test_filename' in kwargs:
            self.test_filename = kwargs['test_filename']
        else:
            self.test_filename = None
        if 'val_filename' in kwargs:
            self.val_filename = kwargs['val_filename']
        else:
            self.val_filename = None

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """

        # transforms = [
        #     tform.MatScipyNeighborList(cutoff=self.cutoff),
        # ]

        # if self.remove_offsets:
        #     transforms.insert(
        #         0,
        #         tform.RemoveOffsets('DS', remove_mean=True, remove_atomrefs=False)
        #     )
        
        transforms = [
            tform.SubtractCenterOfMass(),
            tform.MatScipyNeighborList(cutoff=self.cutoff),
            tform.CastTo32()
            ]

        datamodule = AtomsDataModule(
            datapath=os.path.join(stage, '{}.db'.format(self.database_name)),
            split_file=os.path.join(stage, 'split.npz'),
            format=AtomsDataFormat.ASE,
            load_properties=['DS'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transforms=transforms,
        )

        datamodule.setup()

        # TODO: allow optional loading of train/test/val files by name

        self.train_dataset  = datamodule.train_dataset
        self.test_dataset   = datamodule.test_dataset
        self.val_dataset    = datamodule.val_dataset


    def get_dataloader(self, dataset):
        return AtomsLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                # shuffle=True,
                # pin_memory=self._pin_memory,
            )





