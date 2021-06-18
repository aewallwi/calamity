import tensorflow as tf
import numpy as np
import datetime
from .echo import echo


def simple_cov_matrix(uvdata, baseline_group, ant_dly, dtype=np.float32):
    """Compute simple covariance matrix for subset of baselines in uvdata

    Parameters
    ----------
    uvdata: UVdata object
        uvdata with baselines to produce covariance for.
    baseline_group: tuple of tuples of 2-tuples
        tuple of tuple of 2-tuples where each tuple is a different redundant baseline
        group that contains 2-tuples specifying each baseline.
    ant_dly: float
        intrinsic chromaticity of each antenna element.
    dtype: numpy.dtype
        data type to process and store covariance matrix as.

    Returns
    -------
    cmat: tf.Tensor object
        (Nbls * Nfreqs) x (Nbls * Nfreqs) tf.Tensor object

    """
    ap_data = set(uvdata.get_antpairs())
    uvws = []
    for red_grp in baseline_group:
        ap = red_grp[0]
        if ap in ap_data:
            dinds = uvdata.antpair2ind(red_grp[0])
            uvws.extend(uvdata.uvw_array[dinds])
        else:
            dinds = uvdata.antpair2ind(red_grp[0][::-1])
            uvws.extend(-uvdata.uvw_array[dinds])
    # find differences between all uvws
    uvws = np.asarray(uvws)
    nbls = len(uvws)
    uvws = tf.convert_to_tensor(uvws, dtype=dtype)
    freqs = tf.convert_to_tensor(np.asarray(uvdata.freq_array.squeeze()), dtype)
    absdiff = tf.convert_to_tensor(np.zeros((nbls, nbls), dtype=dtype), dtype=dtype)
    for uvw_ind in range(3):
        uvw_coord = tf.reshape(tf.outer(uvws[:, uvw_ind], freqs / 3e8), (nbls * uvdata.Nfreqs,))
        cg0, cg1 = tf.meshgrid(uvw_coord, uvw_coord, indexing="ij")
        absdiff += np.abs(cg0 - cg1) ** 2.0
    absdiff = absdiff ** 0.5
    cmat = tf.experimental.numpy.sinc(2 * absdiff)
    del absdiff
    fg0, fg1 = tf.reshape(tf.outer(tf.ones(nbls, dytpe=dtype), freqs), (nbls * uvdata.Nfreqs,), indexing="ij")
    cmat = cmat * tf.experimental.sinc(2 * tf.abs(fg0 - fg1) * ant_dly)
    del fg0, fg1
    return cmat


def yield_simple_multi_baseline_model_comps(uvdata, baseline_group, ant_dly, dtype=np.float32, verbose=False):
    """Generate model components for multiple baselines.

    Parameters
    ----------
    uvdata: UVdata object
        uvdata holding data that you will be modeling
    antpairs: tuple of tuples of 2-tuples
        tuple of tuples where each sub-tuple is a redundant group
        that you will be jointly modeling which in turn contains
        2-tuples representing antpairs in each redundant group
    dtype: numpy.dtype, optional
        data-type to use for eigenvalue decomposition
        default is np.float32
    verbose: bool, optional
        text outputs
        default is False.

    Returns
    -------
    multi_baseline_model_comps: dict
        dict with baseline_group as single key and
        (Nbl * Nfreqs) x Ncomponents tf.Tensor array as single value

    """
    cmat = simple_cov_matrix(uvdata, baseline_group, ant_dly, dtype=dtype, verbose=verbose)
    # get eigenvectors
    echo(
        f"{datetime.datetime.now()} Deriving modeling components with eigenvalue decomposition...\n",
        verbose=verbose,
    )
    evals, evecs = tf.linalg.eigh(cmat)
    evals_n = evals.numpy()
    selection = evals_n / evals_n.max() >= evals_n
    evals = evals[selection][::-1]
    evecs = evecs[:, selection][:, ::-1]
    return {baseline_group: evecs.numpy()}
