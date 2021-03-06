import torch
import os

from core import *
from torch_backend import *
from resnet import net_half, net_full

import sys
sys.path.append("../prunedlayersim")
from layer_sim.lottery_ticket.LF_Mask import LF_mask_global as Mask
from layer_sim.utils.pytorch_utils import apply_mask


def train(data_dir, epochs_train=24, batch_size=512, device="cuda:0" if torch.cuda.is_available() else "cpu", net=None, state_load=None, save_file=None, mask_grad=None, verbose=1, half=True):
    dataset = cifar10(root=data_dir)
    lr_schedule=PiecewiseLinear([0, 5, epochs_train], [0, 0.4, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    state_destination = save_file
    #os.path.join(prunedlayersim_root, "models/resnet/resnet_fast.torch")

    # controls

    if state_destination is not None:
        if os.path.isdir(state_destination):
            raise RuntimeError(f"state_destination ({state_destination}) is a dir. Must be a file")
        if os.path.dirname(state_destination) not in (".",""):
            os.makedirs(os.path.dirname(state_destination), exist_ok=True)

    # dataset preparation
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    valid_set = list(zip(*preprocess(dataset['valid'], transforms).values()))

    train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True, half=half)
    valid_batches = DataLoader(valid_set, batch_size, shuffle=False, drop_last=False, half=half)
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size

    if net is None:
        model = net_half if half else net_full
        net = model(device)
        if verbose:
            print("Created new model")
    else:
        if verbose:
            print("Using previously defined model")

    if state_load is not None:
        net.load_state_dict(torch.load(state_load)["state_dict"])
        if verbose:
            print(f"Loading state dict from {state_load}")        


    opts = [SGD(trainable_params(net).values(), {'lr': lr, 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)})]
    logs, state = Table(), {MODEL: net, LOSS: x_ent_loss, OPTS: opts, MASK: mask_grad}

    for epoch in range(epochs_train):
        logs.append(union({'epoch': epoch+1}, train_epoch(state, Timer(torch.cuda.synchronize), train_batches, valid_batches)), verbose=verbose>1)

    final_stats = {"train_acc": logs.log[-1]["train"]["acc"]}
    final_stats["test_acc"] = logs.log[-1]["valid"]["acc"]
    final_stats["train_loss"] = logs.log[-1]["train"]["loss"]
    final_stats["test_loss"] = logs.log[-1]["valid"]["loss"]
    final_stats["state_dict"] = net.state_dict()      


    if state_destination is not None:
        torch.save(final_stats, state_destination)
        if verbose:
            print(f"==> Dumped state_dict & stats to {os.path.abspath(state_destination)}")
    else:
        
        if verbose:
            print("==> state_dict not saved since no viable path was passed as save_file")
        
        return final_stats


if __name__=="__main__":
    ## env vars
    prunedlayersim_root = "../prunedlayersim"
    DATA_DIR = os.path.join(prunedlayersim_root,"data")
    half = False
    save_file = None #prunedlayersim_root + "/models/resnet/resnet_fast_full.torch"
    net = net_full() if not half else net_half()
    mask = None # Mask(net, 0.5, previous_mask=None, layertypes_to_prune=["conv"])
    #apply_mask(net, mask)
    train(DATA_DIR, net=net, save_file=save_file, mask_grad=mask, half=half, verbose=10)
