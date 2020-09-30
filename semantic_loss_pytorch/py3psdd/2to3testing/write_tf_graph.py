import os
from os import path
import tensorflow as tf
import sys


def write_tf_graph(vtree_filename, sdd_filename, numvars, version):
    print("Reading vtree")
    vtree = Vtree.read(vtree_filename)
    manager = SddManager(vtree)
    print("Reading sdd")
    alpha = io.sdd_read(sdd_filename,manager)

    pmanager = PSddManager(vtree)
    print("Creating psdd")
    beta = pmanager.copy_and_normalize_sdd(alpha, vtree)

    print("Building tf graph")
    with tf.Graph().as_default() as graph:
        expected_output = tf.placeholder(tf.float32, [None, numvars], name="input_probabilities")
        # yleaves = tf.unstack(tf.nn.sigmoid(expected_output), axis=1)
        yleaves = tf.unstack(expected_output, axis=1)
        yleaves = [[1.0 - ny, ny] for ny in yleaves]
        tfac = beta.generate_tf_ac(yleaves)
        tf.identity(tfac, name="wmc_output")


    graph_def = graph.as_graph_def()
    graphname = vtree_filename.split("/")[-1].split(".")[0]
    logdir = "py%sgraphs" % version
    print("Saving graph in %s" % logdir)
    tf.io.write_graph(graph_def, logdir=logdir, name="graph_%s.txt" % graphname, as_text=True)




if __name__ == '__main__':
    version = sys.version_info[0]
    print("Running pysdd version made for python %s" % version)
    print("Attemptin import of pysdd (python %s)" % version)
    if version == 3:
        import py3psdd
        from py3psdd import Timer,Vtree,SddManager,SddNode, PSddManager
        from py3psdd import io
        print("Imported %s" % py3psdd.__name__)
    elif version == 2:
        import pypsdd
        from pypsdd import Timer,Vtree,SddManager,SddNode, PSddManager
        from pypsdd import io
        print("Imported %s" % pypsdd.__name__)
    else:
        print("Python runtime version not recognized (must be 2 or 3)")
        exit()
    
    vtree, sdd, numvars = sys.argv[1:]
    write_tf_graph(vtree, sdd, numvars, version)


