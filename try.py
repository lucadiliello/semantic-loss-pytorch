from py3psdd import SddManager, Vtree, io, PSddManager


vtree = Vtree.read("constraint.vtree")
manager = SddManager(vtree)
alpha = io.sdd_read("constraint.sdd", manager)
pmanager = PSddManager(vtree)
psdd = pmanager.copy_and_normalize_sdd(alpha, vtree)

print(psdd.generate_tf_ac_v2)

exit()