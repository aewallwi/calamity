import tensorflow as tf
import scipy.signal as signal
import numpy as np


def _fourier_filter_hash(filter_centers, filter_half_widths, filter_factors, x, w=None, hash_decimal=10, **kwargs):
    """
    Generate a hash key for a fourier filter

    Parameters
    ----------
        filter_centers: list,
                        list of floats for filter centers
        filter_half_widths: list
                        list of float filter half widths (in fourier space)

        filter_factors: list
                        list of float filter factors
        x: the x-axis of the data to be subjected to the hashed filter.
        w: optional vector of float weights to hash to. default, none
        hash_decimal: number of decimals to use for floats in key.
        kwargs: additional hashable elements the user would like to
                include in their filter key.

    Returns
    -------
    A key for fourier_filter arrays hasing the information provided in the args.
    """
    filter_key = (
        ("x:",)
        + tuple(np.round(x, hash_decimal))
        + ("filter_centers x N x DF:",)
        + tuple(np.round(np.asarray(filter_centers) * np.mean(np.diff(x)) * len(x), hash_decimal))
        + ("filter_half_widths x N x DF:",)
        + tuple(
            np.round(
                np.asarray(filter_half_widths) * np.mean(np.diff(x)) * len(x),
                hash_decimal,
            )
        )
        + ("filter_factors x 1e9:",)
        + tuple(np.round(np.asarray(filter_factors) * 1e9, hash_decimal))
    )
    if w is not None:
        filter_key = filter_key + ("weights",) + tuple(np.round(w.tolist(), hash_decimal))
    filter_key = filter_key + tuple([kwargs[k] for k in kwargs])
    return filter_key


def place_data_on_uniform_grid(x, data, weights, xtol=1e-3):
    """If possible, place data on a uniformly spaced grid.

    Given a vector of x-values (x), with data and weights,
    this function determines whether there are gaps in the
    provided x-values that are multiples of the minimum
    distance between x-values or whether any gaps are
    integer multiples of a fundamental grid spacing.
    If there are gaps that are integer multiples of a
    fundamental spacing, this function restores these
    x-values and inserts zero-valued
    data and zero-valued weights at their location,
    returning equally spaced data and weights that are
    effectively flagged at the missing x-values.
    This supports filtering data that was regularly sampled but has
    missing samples due to (for example) correlator dropouts since
    several of our filtering methods (DPSS fits and CLEAN) require data
    to be sampled on an equally spaced grid.

    Parameters
    ----------
    x: array-like,
        array of x-values.
    data: array-like,
        array of y-values.
        Should be the same length as x.
    weights: array-like,
        array of weights.
        Should be the same length as x.
    xtol: float, optional.
        fractional error tolerance to determine if x-values are
        on an incomplete grid.

    Returns
    -------
        xout: array-like
              If the separations on x are multiples of a single underlying minimum unit
              returns x with all multiples of the fundamental unit filled in.
              If x is already uniformly spaced, returns x unchanged. If separations are not
              multiples of fundamental unit, also returns x unchanged.
        yout: array-like
              If the separations on x are multiples of a single underlying minimum unit
              returns y with all multiples of the fundamental unit filled in with zeros.
              If x is already uniformly spaced, returns y unchanged. If separations are not
              multiples of fundamental unit, also returns y unchanged.
        wout: array-like
              If the separations on x are multiples of a single underlying minimum unit
              returns w with all multiples of the fundamental unit filled in with zeros.
              If x is already uniformly spaced, returns w unchanged. If separations are not
              multiples of fundamental unit, also returns w unchanged.
        inserted: array-like
              boolean array indicating which x-values were inserted.
    """
    xdiff = np.diff(x)
    dx = np.abs(np.diff(x)).min() * np.sign(np.diff(x)[0])
    # first, check whether x, y, w already on a grid.
    # if they are, just return them.
    if np.allclose(xdiff, dx, rtol=0, atol=dx * xtol):
        xout = x
        dout = data
        wout = weights
        inserted = np.zeros(len(x), dtype=bool)
        return xout, dout, wout, inserted
    # next, check that the array is not on a grid and if it isn't, return x, y, w
    if not np.allclose(xdiff / dx, np.round(xdiff / dx), rtol=0.0, atol=np.abs(xtol * dx)):
        xout = x
        dout = data
        wout = weights
        inserted = np.zeros(len(x), dtype=bool)
        warn(
            "Data cannot be placed on equally spaced grid! No values inserted.",
            RuntimeWarning,
        )
        return xout, dout, wout, inserted
    # if the array is on a grid, then construct filled in grid.
    grid_size = int(np.round((x[-1] - x[0]) / dx)) + 1
    xout = np.linspace(x[0], x[-1], grid_size)
    dout = np.zeros(grid_size, dtype=np.complex128)
    wout = np.zeros(grid_size, dtype=np.float)
    inserted = np.ones(grid_size, dtype=bool)
    # fill in original data and weights.
    for x_index, xt in enumerate(x):
        output_index = np.argmin(np.abs(xout - xt))
        dout[output_index] = data[x_index]
        wout[output_index] = weights[x_index]
        inserted[output_index] = False

    return xout, dout, wout, inserted


def tensorflow_dpss(
    nf,
    nfw,
    nwindows=None,
    lobe_ordering_like_scipy=False,
    method="sinc_inversion",
    dtype=np.float64,
):
    """
    Tensorflow version of the scipy.signals.windows.dpss

    This recipe is based on the scipy.windows.dpss method whose code is as at
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/windows/windows.py
    We have modified it to use tensorflow methods for spectral decomposition
    so that it is GPU accelerateable.

    Parameters
    ----------
    nf: int
        length of the window
    nfw: float
        standardized half bandwidth. See (NW) argument in scipy.windows.dpss
    nwindows: int, optional
        number of windows to return
        default is None -> return nwindows = nf.
    lobe_ordering_like_scipy: bool, optional
        if True, ensure that the parity of the lobes in each dpss vector matches
        the output from scipy.windows.dpss. Not strictly necessary for many applications
        except for direct comparisons to scipy method.
    method: str, optional
        the method to use for determining dpss windows. Supported are "eigh_tridiagonal"
        which is the method used by scipy.signal.windows.dpss but seems to fail
        with large nf. "eigh_sinc" uses eigen-decomposition of a sinc matrix.
    dtype: numpy dtype
        data type for tensors.
        default is np.float64

    Returns
    -------
    dpss_vectors: np.ndarray
        nwindows X nf array of dpss windows arranged in non-descending order of
        eigenvalues with the sinc matrix.

    """
    # extend by 1
    if nwindows is None:
        nwindows = nf + 1
    # use sinc method.
    # since many eigenvalues will be close to 1, the specific eigenvectors
    # dont necessarily agree with the tridiagonal method but they are spanning
    # the same space and they are orthonormal.
    if method == "eigh_sinc":
        xg, yg = np.meshgrid(np.arange(nf), np.arange(nf))
        dxy = tf.convert_to_tensor(xg - yg, dtype=dtype)
        smat = tf.experimental.numpy.sinc(2 * nfw * dxy / nf) * 2 * nfw / nf
        _, dpss_vecs = tf.linalg.eigh(smat)
    # perform symmetric tri-diagonal eigen-decomposition on the GPU (if available)
    # this is the method used by scipy.windows.dpss but now with tensorflow.
    # https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/windows/windows.py
    # for some reason it gives errors for large numbers of rows / columns
    # which often come up with fringe-rate filtering.
    # so made eigh_sinc the default (although it seems like in principal its slower)
    elif method == "eigh_tridiagonal":
        nidx = np.arange(nf)
        d = ((nf - 1 - 2 * nidx) / 2.0) ** 2 * np.cos(2 * np.pi * nfw / nf)
        e = nidx[1:] * (nf - nidx[1:]) / 2.0
        _, dpss_vecs = tf.linalg.eigh_tridiagonal(
            tf.convert_to_tensor(d, dtype=dtype),
            tf.convert_to_tensor(e, dtype=dtype),
            eigvals_only=False,
        )
    dpss_vecs = tf.transpose(dpss_vecs[:, ::-1])
    # the following code is to make the ordering of the lobes agree with the scipy method convention
    # which is in turn consistent with Percival and Walden 1993 pg 379.
    # this option is really only necessary for testing where we want to directly
    # compare outputs from scipy.windows.dpss with outputs from tensorflow_dpss
    if lobe_ordering_like_scipy:
        dpss_vecs = dpss_vecs.numpy()
        fix_even = tf.reduce_sum(dpss_vecs[::2], axis=1) < 0
        for i, f in enumerate(fix_even):
            if f:
                dpss_vecs[2 * i] *= -1

        thresh = max(1e-7, 1.0 / nf)
        for i, w in enumerate(dpss_vecs[1::2]):
            if w[w * w > thresh][0] < 0:
                dpss_vecs[2 * i + 1] *= -1
        dpss_vecs = tf.convert_to_tensor(dpss_vecs, dtype=dtype)

    return dpss_vecs[:nwindows]


def dpss_operator(
    x,
    filter_centers,
    filter_half_widths,
    cache=None,
    eigenval_cutoff=1e-10,
    xc=None,
    hash_decimal=10,
    xtol=1e-3,
    use_tensorflow=False,
    lobe_ordering_like_scipy=False,
    tf_method="eigh_sinc",
    dtype=np.float64,
):
    """
    Calculates DPSS operator with multiple delay windows to fit data. Frequencies
    must be equally spaced (unlike Fourier operator). Users can specify how the
    DPSS series fits are cutoff in each delay-filtering window with one (and only one)
    of three conditions: eigenvalues in sinc matrix fall below a thresshold (eigenval_cutoff),
    user specified number of DPSS terms (nterms), xor the suppression of fourier
    tones at the filter edge by a user specified amount (edge_supression).

    Parameters
    ----------
    x: array-like
        x values to evaluate operator at
    filter_centers: array-like
        list of floats of centers of delay filter windows in nanosec
    filter_half_widths: array-like
        list of floats of half-widths of delay filter windows in nanosec
    cache: dictionary, optional
        dictionary for storing operator matrices with keys
        tuple(x) + tuple(filter_centers) + tuple(filter_half_widths)\
         + (series_cutoff_name,) = tuple(series_cutoff_values)
    eigenval_cutoff: list of floats, optional
        list of sinc matrix eigenvalue cutoffs to use for included dpss modes.
        default is 1e-10
    xc: float optional
    hash_decimal: number of decimals to round for floating point dict keys.
    xtol: fraction of average diff that the diff between all x-values must be within
          the average diff to be considered
          equally spaced. Default is 1e-3
    use_tensorflow: bool, optional
        if True, use tensorflow matrix inversion to derive dpss operator.
    tensorflow_lobe_ordering_like_scipy: bool, optional
        if True, set the sign of the first lobes of each dpss vector to be
        consistent with the ordering in scipy. This is for testing purposes only.
        default is False.
    dtype: np.dtype
        precision for tensorflow operations. Used if use_tensorflow is True.
    Returns
    ----------
    2-tuple
    First element:
        Design matrix for DPSS fitting.   Ndata x (Nfilter_window * nterm)
        transforming from DPSS modes to data.
    Second element:
        list of integers with number of terms for each fourier window specified by filter_centers
        and filter_half_widths
    """
    if isinstance(eigenval_cutoff, float):
        eigenval_cutoff = [eigenval_cutoff for fw in range(len(filter_centers))]
    if cache is None:
        cache = {}
    opkey = _fourier_filter_hash(
        filter_centers=filter_centers,
        filter_half_widths=filter_half_widths,
        filter_factors=[0.0],
        crit_name="eigenval_cutoff",
        x=x,
        w=None,
        hash_decimal=hash_decimal,
        use_tensorflow=use_tensorflow,
        label="dpss_operator",
        crit_val=tuple(eigenval_cutoff),
    )
    if not opkey in cache:
        # try placing x on a uniform grid.
        # x is a version of x with the in-between grid values filled in and inserted is a boolean vector
        # set to True wherever a value for x was inserted and False otherwise.
        x, _, _, inserted = dpss.place_data_on_uniform_grid(x, np.zeros(len(x)), np.ones(len(x)))
        # if this is not successful, then throw a value error..
        if not np.allclose(
            np.diff(x),
            np.median(np.diff(x)),
            rtol=0.0,
            atol=np.abs(xtol * np.median(np.diff(x))),
        ):
            # for now, don't support DPSS iterpolation unless x is equally spaced.
            # In principal, I should be able to compute off-grid DPSS points using
            # the fourier integral of the DPSWF
            raise ValueError("x values must be equally spaced for DPSS operator!")
        nf = len(x)
        df = np.abs(x[1] - x[0])
        xg, yg = np.meshgrid(x, x)
        if use_tensorflow:
            xg = tf.convert_to_tensor(xg, dtype=dtype)
            yg = tf.convert_to_tensor(yg, dtype=dtype)
        if xc is None:
            xc = x[nf // 2]
        # determine cutoffs
        dpss_vectors = []
        evals_precomp = []
        for fw in filter_half_widths:
            if not use_tensorflow:
                dpss_vecs = windows.dpss(nf, nf * df * fw, nf)
                dpss_vectors.append(dpss_vecs)
                # now find concentration
                smat = np.sinc(2 * fw * (xg - yg)) * 2 * df * fw
                evals = np.sum((smat @ np.transpose(dpss_vecs)) * np.transpose(dpss_vecs), axis=0)
                evals_precomp.append(evals)
            else:
                dpss_vecs = tensorflow_dpss(
                    nf,
                    nf * df * fw,
                    nf,
                    lobe_ordering_like_scipy=lobe_ordering_like_scipy,
                    method=tf_method,
                    dtype=dtype,
                )
                smat = tf.experimental.numpy.sinc(2 * fw * (xg - yg)) * 2 * df * fw
                evals = tf.reduce_sum((smat @ tf.transpose(dpss_vecs)) * tf.transpose(dpss_vecs), axis=0)
                evals_precomp.append(evals)
                dpss_vectors.append(
                    dpss_vecs
                )  # eigenvals and eigenvectors are in non-decreasing order. Switch to non-increasing order.
                evals_precomp.append(evals)
        nterms = []
        for fn, fw in enumerate(filter_half_widths):
            if not use_tensorflow:
                nterms.append(np.max(np.where(evals_precomp[fn] >= eigenval_cutoff[fn])))
            else:
                nterms.append(np.max(np.where(evals_precomp[fn].numpy() >= eigenval_cutoff[fn])))
        # next, construct A matrix.
        amat = []
        for fn, (fc, fw, nt) in enumerate(zip(filter_centers, filter_half_widths, nterms)):
            if use_tensorflow:
                dpss_vecs = dpss_vectors[fn].numpy()[:nt].T
                ygn = yg.numpy()
            else:
                dpss_vecs = dpss_vectors[fn][:nt].T
                ygn = yg
            amat.append(np.exp(2j * np.pi * (ygn[:, :nt] - xc) * fc) * dpss_vecs)
        if len(amat) > 1:
            amat = np.hstack(amat)
        else:
            amat = amat[0]
        # we used the regularly spaced inserted grid to generate our fitting basis vectors
        # but we dont need them for the actual fit.
        # so here we keep only the non-inserted rows of the design matrix.
        amat = amat[~inserted, :]
        cache[opkey] = (amat, nterms)
    return cache[opkey]
