import tqdm
import tqdm.notebook as tqdm_notebook
import numpy as np

PBARS = {True: tqdm_notebook.tqdm, False: tqdm.tqdm}


def echo(message, verbose=True):
    if verbose:
        print(message)


def select_baselines(uvdata, bllen_min=0.0, bllen_max=np.inf, bl_ew_min=0.0, ex_ants=None, select_ants=None):
    """ """
    if ex_ants is None:
        ex_ants = []
    ex_ants = set(ex_ants)
    antpos, antnums = uvdata.get_ENU_antpos(pick_data_ants=True)
    if select_ants is None:
        select_ants = set(antnums)
    antpairs = uvdata.get_antpairs()
    posdict = {an: ap for an, ap in zip(antnums, antpos)}
    antpairs_to_keep = []
    for ap in antpairs:
        blvec = posdict[ap[0]] - posdict[ap[1]]
        bllen = np.linalg.norm(blvec)
        if (
            bllen >= bllen_min
            and bllen <= bllen_max
            and np.abs(blvec[0]) > bl_ew_min
            and ap[0] not in ex_ants
            and ap[1] not in ex_ants
            and ap[0] in select_ants
            and ap[1] in select_ants
        ):
            antpairs_to_keep.append(ap)
    uvdata.select(bls=antpairs_to_keep, inplace=True)


# def fitting_group_nested_tuple_to_str(fit_grp_tuple):
#    """
#    Convert a fitting group nested tuple
#    to a string / legal python variable name
#    that can be used to key an npz file.

#    Parameters
#    ----------
#    fit_grp_tuple: nested tuple
#        tuple of tuple of int 2-tuples representing fitting groups.
#
#
#    """
#    output_str = ''
#    for redgrp in fit_grp_tuple:
#        output_str += 'rg'
#        for ap in redgrp:
#            output_str += f'{ap[0]}x{ap[1]}and'
#
# def fitting_group_str_to_nested_tuple():
#    """
#
#    """
