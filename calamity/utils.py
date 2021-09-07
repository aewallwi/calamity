import tqdm
import tqdm.notebook as tqdm_notebook

PBARS = {True: tqdm_notebook.tqdm, False: tqdm.tqdm}


def echo(message, verbose=True):
    if verbose:
        print(message)

def select_baselines_on_length(uvdata, bllen_min=0., bllen_max=np.inf, bl_ew_min=0.0):
    """
    """
    antpos, antnums = uvdata.get_ENU_antpos(pick_data_ants=True)
    antpairs = uvdata.get_antpairs()
    posdict = {an: ap for an, ap in zip(antnums, antpairs)}
    antpairs_to_select = []
    for ap in antpairs:
        blvec = posdict[ap[0]] - posdict[ap[1]]
        bllen = np.linalg.norm(blvec)
        if bllen >= bllen_min and bllen <= bllen_max and np.abs(blvec[0]) > bl_ew_min:
            antpairs_to_keep.append(ap)
    uvdata.select(bls=antpairs_to_keep, inplace=True)



#def fitting_group_nested_tuple_to_str(fit_grp_tuple):
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
#def fitting_group_str_to_nested_tuple():
#    """
#
#    """
