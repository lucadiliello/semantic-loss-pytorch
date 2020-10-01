import os
import json
from functools import reduce

import torch
from torch.nn.modules.loss import _Loss
from semantic_loss_pytorch.py3psdd import Vtree, SddManager, PSddManager, io

# For numerical stability
EPSILON = 1e-9


class SemanticLoss(_Loss):
    """
    Module containing the semantic loss.
    To use the loss, simply pass the relative vtree and sdd files.
    The forward method is mostly copied by
    https://github.com/UCLA-StarAI/Semantic-Loss/blob/master/complex_constraints/compute_mpe.py,
    from the semantic loss paper, currently it's basically the same class except for names changed to my liking
    and different imports and comments.
    """

    @staticmethod
    def _import_psdd(sdd_file: str, vtree_file: str) -> PSddManager:
        """
        Given a constraint_name, assert the existence and look for the related .vtree and .sdd files.
        The vtree and sdd are loaded and used to instantiate the psdd, which is then returned.
        :param sdd_file: Name of the `sdd` file to use.
        :param vtree_file: Name of the `vtree` file to use.
        """

        assert os.path.isfile(sdd_file), f"{sdd_file} is not a file."
        assert os.path.isfile(vtree_file), f"{vtree_file} is not a file."

        # load vtree and sdd files and construct the PSDD
        vtree = Vtree.read(vtree_file)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_file, manager)
        pmanager = PSddManager(vtree)
        psdd = pmanager.copy_and_normalize_sdd(alpha, vtree)

        return psdd

    def __init__(self, sdd_file: str, vtree_file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # instantiate the psdd
        self.psdd = SemanticLoss._import_psdd(sdd_file, vtree_file)

    def forward(
        self,
        logits: torch.FloatTensor = None,
        probabilities: torch.FloatTensor = None,
        output_wmc:bool = False,
        output_wmc_per_sample: bool = False
    ) -> torch.FloatTensor:
        """
        Returns the semantic loss related to the instance of this class, using the `x` input.
        If input are logits, the sigmoid function is applied to the input.
        
        :param x: input tensor that will be interpreted as probabilities or logits (input_are_logits=True)
        :return: the weighted model count for the input tensor `x` with respect to the psdd
        """
        if (logits is None) == (probabilities is None):
            raise ValueError("Only logits or probabilities can be provided, neither both nor none")

        # set logits to probabilities
        if logits is not None:
            probabilities = torch.sigmoid(logits)

        # need to reshape as a 1d vector of variables for each sample, needed by psdd for the torch AC
        batch_size, *other_dims = probabilities.size()
        total_variables = reduce(lambda x, y: x * y, other_dims)
        probs_as_vector = torch.reshape(probabilities, (batch_size, total_variables))

        wmc_per_sample = self.psdd.generate_pt_ac_v2(probs_as_vector)
        wmc = torch.mean(wmc_per_sample)
        loss = -torch.log(wmc)

        outputs = (loss,)
        if output_wmc:
            outputs = outputs + (wmc,)
        if output_wmc_per_sample:
            outputs = outputs + (wmc_per_sample,)
        
        return outputs if len(outputs) > 1 else outputs[0]
