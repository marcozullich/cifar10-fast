import sys
import os
import torch

from resnet import net_half as model
from train import train

sys.path.append("../prunedlayersim")
from layer_sim.lottery_ticket.LF_Mask import LF_mask_global as Mask, prop_parameters_mask
from layer_sim.utils.pytorch_utils import apply_mask

def IMP(num_ite, fully_trained_state, pr_rate=0.2, device="cuda:0" if torch.cuda.is_available() else "cpu", intermediate_dumps="IMP_checkpoint_{}.torch", verbose=1, resume_ite=None, data_dir="../prunedlayersim/data", **kwargs):
    '''
    SUPPORTS ONLY
        Learning Rate Rewind
    WITH
        Global Masking
    '''

    if fully_trained_state is not None and resume_ite is not None:
        raise RuntimeError(f"fully_trained_state ({fully_trained_state}) and resume_ite ({resume_ite}) are both specified. Only one of the two must be specified (fully_trained_state -> start IMP from a complete network; resume_ite -> continue IMP from a pretrained IMP state, needs no fully_trained_state to start with).")

    if fully_trained_state is None and resume_ite is None:
        raise RuntimeError("fully_trained_state and resume_ite are both set to None. One of the two needs to be valued in order for IMP to be carried out.")

    if not isinstance(num_ite, int):
        raise TypeError(f"num_ite {(num_ite)} must be an int.")

    # create dump folder
    dir_interm_dump = os.path.dirname(intermediate_dumps)
    if dir_interm_dump not in ("","."):
        os.makedirs(dir_interm_dump, exist_ok=True)
    del dir_interm_dump

    # create model
    net = model(device)

    # load state_dict -> different paths if starting from scratch or resuminig previous iteration
    net_state_name = fully_trained_state or intermediate_dumps.format(resume_ite)

    net_state = torch.load(net_state_name)
    del net_state_name
    net.load_state_dict(net_state["state_dict"])

    mask = net_state.get("mask") # None if no mask 
    
    if verbose > 0:
            print(f"Loaded state dict from {os.path.abspath}")
            if verbose > 1:
                print(f"   Stats from loaded model: test acc {net_state["test_acc"]}; train acc {net_state["train_acc"]}; test loss {net_state["test_loss"]}; train loss {net_state["train_loss"]}.")
    
    del net_state

    # increase ite_resume
    if ite_resume is not None:
        ite_resume += 1
    else:
        ite_resume = 0

    # proper IMP
    for ite in range(ite_resume, num_ite):
        if verbose > 0:
            print(f"**Iteration of IMP {num_ite+1}")

        # calculate new mask
        mask = Mask(net, 1 - pr_rate, previous_mask=mask, layertypes_to_prune="conv")

        if verbose > 1:
            p = prop_parameters_mask(mask)
            print(f"    Proportion of parameters in mask = {(prop_parameters_mask)}")

        # apply mask to state_dict
        apply_mask(net, mask)

        # retrain passing mask
        train_state = train(data_dir, net=net, mask_grad=mask, verbose=verbose-2, **kwargs)

        train_state["mask"] = mask

        torch.save(train_state, intermediate_dumps.format(ite))
        if verbose > 1:
            print(f"    Dumped state_dict, stats, and mask to {intermediate_dumps.format(ite)}")

if __name__=="__main__":





    



