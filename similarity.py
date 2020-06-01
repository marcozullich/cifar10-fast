import torch
import os
from resnet import net_full

import sys
sys.path.append("../prunedlayersim")

from layer_sim.utils.svcca_utils import get_layers_batches
from layer_sim.CIFAR.cifar10_loader import datasetLoader
from layer_sim.Grad_CKA.sim import SimIndex
from layer_sim.utils.reshape import reshape_layer, ReshapingTechnique
from layer_sim.utils.svd import svd_reduce

os.makedirs("SIMS", exist_ok=True)

net = net_full()
loader,_,_ = datasetLoader("minimal", False).get_data(list(range(10)), 512, 1000, seed=1, store_path="../prunedlayersim/data")
keys_select = ("prep/relu", "layer1/relu", "layer1/pool", "layer2/relu", "layer2/pool", "layer3/relu", "layer3/pool", "linear")

print("Evaluating complete model...")

net.load_state_dict(torch.load("../prunedlayersim/models/resnet/resnet_fast_full.torch")["state_dict"])

compl, _ = get_layers_batches(net, loader, "cuda", limit_dataset_numerosity=5000, model_type="RESNET", resnet_keys_select=keys_select)

print("Evaluating pruned model...")

net.load_state_dict(torch.load("../prunedlayersim/models/resnet/IMP/0/IMP_checkpoint_9.torch")["state_dict"])

pruned, _ = get_layers_batches(net, loader, "cuda", limit_dataset_numerosity=5000, model_type="RESNET", resnet_keys_select=keys_select)

print("Applying reshapes...")

compl2 = [reshape_layer(c, ReshapingTechnique(1)) for c in compl]
pruned2 = [reshape_layer(c, ReshapingTechnique(1)) for c in pruned]

compl = [c.view(c.shape[0],-1) for c in compl]
pruned = [c.view(c.shape[0],-1) for c in pruned]

print("Getting kernels...")

compl_k = [c@c.T for c in compl]
pruned_k = [c@c.T for c in pruned]

ind = SimIndex()

print("    CKA...")

CKA = [ind.cka(k, kk) for k, kk in zip(compl_k, pruned_k)]

torch.save(CKA, "SIMS/CKA.torch")

print("Saved\n    NBS...")

NBS = [ind.nbs(k, kk) for k, kk in zip(compl_k, pruned_k)]

torch.save(NBS, "SIMS/NBS.torch")

print("Saved\nApplying SVCCA preprocessing...")

compl2 = [svd_reduce(c).T for c in compl2]
pruned2 = [svd_reduce(c).T for c in pruned2]

SVCCA = [ind.mean_cca(c,p) for c, p in zip(compl2, pruned2)]

print("Done.")



