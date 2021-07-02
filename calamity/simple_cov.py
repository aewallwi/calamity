import tensorflow as tf
import numpy as np
import datetime
from .utils import echo


def simple_cov_matrix(
    blvecs, freqs, ant_dly=0.0, horizon=1.0, offset=0.0, min_dly=0.0, dtype=np.float32, use_tensorflow=False
):
    """Compute simple covariance matrix for subset of baselines in uvdata

    Parameters
    ----------
    blvecs: list of np.ndarrays
        list of length-3 np.ndarrays representing ENH baseline vectors.
    freqs: array-like
        array of frequencies (Hz) sampled by interferometer.
    ant_dly: float
        intrinsic chromaticity of each antenna element, manifested in a
        multiplicative sinc matrix sinc(2 pi tau_ant * (nu_1 - nu_0))
    horizon: float, optional
        fraction of horizon for beam to extend to in analytic cov matrix.
        default is 1.0
    offset: float, optional
        additional offset causing additional decorrelation between antennas
        (units of ns).
        default is 0.0
    min_dly: float, optional
        minimum decorrelation delay between points in uv plane. Net effect should
        be some additional modes and signal loss.
        default is 0.0
    dtype: numpy.dtype
        data type to process and store covariance matrix as.
    use_tensorflow: bool, optional
        if True, use tensorflow for multi-baseline modeling matrix operations.
        Recommended only for machines with GPUs.

    Returns
    -------
    cmat: tf.Tensor object (if use_tensorflow is True) or np.ndarray(use_tensorflow is False)
        (Nbls * Nfreqs) x (Nbls * Nfreqs) tf.Tensor object or np.ndarray with dtype dtype


    """
    nbls = len(blvecs)
    uvws = np.asarray(blvecs, dtype=dtype)
    freqs = np.asarray(freqs, dtype=dtype)
    nfreqs = len(freqs)
    freqs = freqs.astype(dtype)
    absdiff = np.zeros((nbls * nfreqs, nbls * nfreqs), dtype=dtype)
    if use_tensorflow:
        freqs = tf.convert_to_tensor(freqs, dtype)
        absdiff = tf.convert_to_tensor(absdiff, dtype=dtype)
    for uvw_ind in range(3):
        if use_tensorflow:
            uvw_coord = tf.reshape(tf.experimental.numpy.outer(uvws[:, uvw_ind], freqs / 3e8), (nbls * nfreqs,))
            cg0, cg1 = tf.meshgrid(uvw_coord, uvw_coord, indexing="ij")
            absdiff += tf.math.square(tf.math.abs(cg0 - cg1) * horizon)
        else:
            uvw_coord = np.reshape(np.outer(uvws[:, uvw_ind], freqs / 3e8), (nbls * nfreqs,))
            cg0, cg1 = np.meshgrid(uvw_coord, uvw_coord, indexing="ij")
            absdiff += np.abs(cg0 - cg1) ** 2.0
    if use_tensorflow:
        absdiff = tf.math.sqrt(absdiff) * horizon
        fvals = tf.reshape(tf.experimental.numpy.outer(tf.ones(nbls, dtype=dtype), freqs), (nbls * nfreqs,))
        fg0, fg1 = tf.meshgrid(fvals, fvals, indexing="ij")
        dfg = tf.abs(fg0 - fg1)
    else:
        absdiff = np.sqrt(absdiff) * horizon
        fvals = np.reshape(np.outer(np.ones(nbls, dtype=dtype), freqs), (nbls * nfreqs,))
        fg0, fg1 = np.meshgrid(fvals, fvals, indexing="ij")
        dfg = np.abs(fg0 - fg1)

    del fg0, fg1
    absdiff += dfg * offset / 1e9
    if use_tensorflow:
        cmat = tf.experimental.numpy.sinc(2 * tf.maximum(absdiff, min_dly * dfg / 1e9))
    else:
        cmat = np.sinc(2 * np.maximum(min_dly * dfg / 1e9, absdiff))
    del absdiff
    if use_tensorflow:
        cmat = cmat * tf.experimental.numpy.sinc(2 * dfg * ant_dly)
    else:
        cmat = cmat * np.sinc(2 * dfg * ant_dly)
    del dfg
    return cmat


def yield_simple_multi_baseline_model_comps(
    blvecs,
    freqs,
    ant_dly=0.0,
    horizon=1.0,
    offset=0.0,
    min_dly=0.0,
    dtype=np.float32,
    verbose=False,
    use_tensorflow=False,
    eigenval_cutoff=1e-10,
):
    """Generate model components for multiple baselines.

    Parameters
    ----------
    blvecs: list of len-3 numpy.ndarrays
      list of baseline vectors ENH
    ant_dly: float, optional
        intrinsic chromaticity of each antenna element, manifested in a
        multiplicative sinc matrix sinc(2 pi tau_ant * (nu_1 - nu_0))
        default is 0.
    freqs: array-like
      array of frequencies sampled by interferometer (MHz).
    horizon: float, optional
        fraction of horizon for beam to extend to in analytic cov matrix.
        default is 1.0
    offset: float, optional
        additional offset causing additional decorrelation between antennas
        (units of ns).
        default is 0.0
    min_dly: float, optional
        minimum decorrelation delay between points in uv plane. Net effect should
        be some additional modes and signal loss.
        default is 0.0
    dtype: numpy.dtype, optional
        data-type to use for eigenvalue decomposition
        default is np.float32
    verbose: bool, optional
        text outputs
        default is False.
    use_tensorflow: bool, optional
        if True, use tensorflow for multi-baseline modeling matrix operations.
        Recommended only for machines with GPUs.
    eigenval_cutoff: float, optional
        threshold of eigenvectors to include in modeling components.

    Returns
    -------
    multi_baseline_model_comps: np.ndarray
        (Nbl * Nfreqs) x Ncomponents numpy.ndarray containing eigenvectors
        with eigenvalues above cutoff threshold.

    """
    cmat = simple_cov_matrix(
        blvecs,
        freqs,
        ant_dly=ant_dly,
        horizon=horizon,
        offset=offset,
        min_dly=min_dly,
        dtype=dtype,
        verbose=verbose,
        use_tensorflow=use_tensorflow,
    )
    # get eigenvectors
    echo(
        f"{datetime.datetime.now()} Deriving modeling components with eigenvalue decomposition...\n",
        verbose=verbose,
    )
    if use_tensorflow:
        evals, evecs = tf.linalg.eigh(cmat)
        evals = evals.numpy()
        evecs = evecs.numpy()
    else:
        evals, evecs = np.linalg.eigh(cmat)

    selection = evals / evals.max() >= eigenval_cutoff
    evals = evals[selection][::-1]
    evecs = evecs[:, selection][:, ::-1]
    return evecs
