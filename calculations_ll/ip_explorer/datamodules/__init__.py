from .schnet import SchNetDataModule
from .schnetDS import SchNetDSDataModule
# from .nequip import NequIPDataModule nie su nainstalovane v systeme a tak nefunguju
# from .ace import ACEDataModule
# from .mace import MACEDataModule
from .ase import ASEDataModule
from .test import TestDataModule


implemented_datamodules = {
    'test':         TestDataModule,
    'schnet':       SchNetDataModule,
    'painn':        SchNetDataModule,
    'schnetDS':     SchNetDSDataModule,
    # 'nequip':       NequIPDataModule,
    # 'allegro':      NequIPDataModule,
    # 'ace':          ACEDataModule,
    # 'mace':         MACEDataModule,
    'valle-oganov': ASEDataModule,
    'vgop':         ASEDataModule,
    'soap':         ASEDataModule,
}

def get_datamodule_wrapper(datamodule_type):
    global implemented_datamodules

    if datamodule_type not in implemented_datamodules:
        raise RuntimeError("Wrapper for datamodule type '{}' has not been implemented.".format(datamodule_type))

    return implemented_datamodules[datamodule_type]
