import os
import json
from functools import reduce

import torch
from py3psdd import Vtree, SddManager, PSddManager, io

# For numerical stability
EPSILON = 1e-9


class SemanticLoss(Loss):
    """
    Module containing the semantic loss.
    To use the loss, simply pass the relative vtree and sdd files.
    The forward method is mostly copied by
    https://github.com/UCLA-StarAI/Semantic-Loss/blob/master/complex_constraints/compute_mpe.py,
    from the semantic loss paper, currently it's basically the same class except for names changed to my liking
    and different imports and comments.
    """

    @staticmethod
    def _import_psdd(sdd_file: str, vtree_file: str):
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

    def __init__(self, sdd_file, vtree_file, *args, input_are_logits=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_are_logits = input_are_logits
        # instantiate the psdd
        self.psdd = SemanticLoss._import_psdd(sdd_file, vtree_file)

    def forward(self, x):
        """
        Returns the semantic loss related to the instance of this class, using the `x` input.
        If input are logits, the sigmoid function is applied to the input.
        
        :param x: input tensor that will be interpreted as probabilities or logits (input_are_logits=True)
        :return: the weighted model count for the input tensor `x` with respect to the psdd
        """

        # set values to probabilities
        toprobs = x

        if self.input_are_logits:
            toprobs = torch.nn.functional.sigmoid(x)

        # need to reshape as a 1d vector of variables for each sample, needed by psdd for the tf AC
        batch_size = toprobs.shape[0]
        total_variables = reduce(lambda x, y: x * y, toprobs.shape[1:])
        probs_as_vector = torch.reshape(toprobs, (batch_size, total_variables))
        
        wmc_per_sample = psdd.generate_tf_ac_v2(probs_as_vector, self.experiment["BATCH_SIZE"])

        
        self.logger.info("Semantic loss wmc of shape %s" % wmc_per_sample.shape)
        
        wmc = tf.reduce_mean(wmc_per_sample)
        self.logger.info("Semantic loss reduced wmc of shape %s" % wmc.shape)
        
        semantic_loss_pre_timing = -tf.log(wmc)
        semantic_loss = self._time_semantic_loss(graph_nodes, semantic_loss_pre_timing)

        nodes["G_loss"] = semantic_loss  # needed cos its expected by the trainer
        nodes[self.constraint_name] = semantic_loss_pre_timing
        nodes[self.constraint_name + "_wmc"] = wmc
        nodes[self.constraint_name + "_wmc_per_sample"] = wmc_per_sample
        nodes_to_log = {**nodes}
        nodes_to_log.pop(self.constraint_name + "_wmc_per_sample")

        del psdd
        return nodes, nodes_to_log, dict()

