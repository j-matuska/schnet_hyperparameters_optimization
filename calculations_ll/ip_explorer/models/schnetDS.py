from .base import PLModelWrapper

import os
import shutil
import torch
import numpy as np

from schnetpack.representation import SchNet
import schnetpack.properties as structure


class SchNetDSModelWrapper(PLModelWrapper):
    """
    A wrapper for a SchNet model. Assumes that `model_path` contains a model
    checkpoint file with the name 'best_model'
    """
    def __init__(self, model_dir, **kwargs):
        if 'representation_type' in kwargs:
            self.representation_type = kwargs['representation_type']
        else:
            self.representation_type = 'node'


        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = torch.load(
            os.path.join(model_path, 'best_model'),
            map_location=torch.device('cpu')
        )


    def compute_loss(self, batch):
        true_eng = batch['DS']
        # true_fcs = batch['DS']*0.0

        results = self.model.forward(batch)

        pred_eng = results['DS']
        # pred_fcs = results['DS']*0.0

        ediff = (pred_eng - true_eng).detach().cpu().numpy()
        # fdiff = (pred_fcs - true_fcs).detach().cpu().numpy()

        return {
            'DS': np.mean(ediff**2),
            #'force':  np.mean(fdiff**2),
            'batch_size': batch['DS'].shape[0],
            'natoms': int(sum(batch['_n_atoms']).detach().cpu().numpy()),
        }

    def compute_energies(self, batch):
        true_eng = batch['DS']
        results = self.model.forward(batch)
        pred_eng = results['DS']
        
        return {
            'true_energies': true_eng,
            'pred_energies': pred_eng,
        }


    # def compute_energies_and_forces(self, batch):
    #     true_eng = batch['energy']/batch['_n_atoms']
    #     true_fcs = batch['forces']

    #     results = self.model.forward(batch)

    #     pred_eng = results['energy']/batch['_n_atoms']
    #     pred_fcs = results['forces']

    #     return {
    #         'true_energies': torch.Tensor(true_eng),
    #         'pred_energies': torch.Tensor(pred_eng),
    #         'true_forces': true_fcs,
    #         'pred_forces': pred_fcs,
    #     }

    def aggregate_energies(self, step_outputs):
        # On each worker, compute the per-structure average representations
        true_energies = []
        pred_energies = []

        for s in step_outputs:
            true_energies.append(s['true_energies'])
            pred_energies.append(s['pred_energies'])


        true_energies = torch.cat(true_energies)
        pred_energies = torch.cat(pred_energies)


        # Now gather everything
        true_energies = self.all_gather(true_energies)
        pred_energies = self.all_gather(pred_energies)


        # Reshape to remove the num_processes dimension.
        # NOTE: order is likely not going to match dataloader order
        true_energies = torch.flatten(true_energies, 0, -1)
        pred_energies = torch.flatten(pred_energies, 0, -1)


        self.results['true_energies'] = true_energies.detach().cpu().numpy()
        self.results['pred_energies'] = pred_energies.detach().cpu().numpy()


    def compute_atom_representations(self, batch):

        # remember: .forward() overwrites the ['energy'] key
        true_eng = (batch['DS']).clone()

        out = self.model.forward(batch)

        with torch.no_grad():
            representations = []

            if self.representation_type in ['node', 'both']:
                representations.append(batch['scalar_representation'])

            if self.representation_type in ['edge', 'both']:

                if isinstance(self.model.representation, SchNet):
                    x = batch['scalar_representation']
                    z = x.new_zeros((batch['scalar_representation'].shape[0], self.model.radial_basis.n_rbf))

                    idx_i = batch[structure.idx_i]
                    idx_j = batch[structure.idx_j]

                    z.index_add_(0, idx_i, batch['distance_representation'])
                    z.index_add_(0, idx_j, batch['distance_representation'])
                else:  # PaiNN
                    z = batch['vector_representation']
                    z = torch.mean(z, dim=1)  # average over cartesian dimension

                representations.append(z)

        representations = torch.cat(representations, dim=1)

        true_eng = (batch['DS']).clone()
        per_atom_energies = torch.cat([
            true_eng.new_ones(n)*e for n,e in zip(batch['_n_atoms'], true_eng)
        ])

        return {
            'representations': representations,
            'representations_splits': batch['_n_atoms'].detach().cpu().numpy().tolist(),
            'representations_energy': true_eng,
        }



    def copy(self, model_path):
        shutil.copyfile(
            os.path.join(model_path, 'best_model'),
            os.path.join(os.getcwd(), 'best_model'),
        )


