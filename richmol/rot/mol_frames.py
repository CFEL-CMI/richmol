import numpy as np
import re


_symm_tol = 1e-14
_small = abs(np.finfo(float).eps)*10
_large = abs(np.finfo(float).max)


_frames = dict()
_opers = dict()

def register_frame(func):
    _frames[func.__name__] = func
    return func

def register_oper(func):
    _opers[func.__name__] = func
    return func


def orthogonality_check(frame, mat):
    imat = np.dot(mat, mat.T)
    if np.any(np.abs(imat - np.eye(3, dtype=np.float64)) > _small):
        raise ValueError(f"'{frame}' frame rotation matrix is not orthogonal") from None
    return mat


def rotmat(frame, *args, **kwargs):
    """Returns rotation matrix to molecular frame specified by 'frame'"""
    if frame in _frames:
        return orthogonality_check(frame, _frames[frame](*args, **kwargs))
    else:
        if re.match(r'(\w+)\((\w+)\)', frame):
            op = re.match('(\w+)\((\w+)\)', frame).group(1)
            tens = re.match('(\w+)\((\w+)\)', frame).group(2)
        else:
            op = "I"
            tens = frame
        if not any(hasattr(arg, tens) for arg in args):
            raise TypeError(f"tensor '{tens}' is not available") from None
        for arg in args:
            if hasattr(arg, tens):
                return orthogonality_check(frame, rotmat_from_tens(tens, getattr(arg, tens), op))
        raise TypeError(f"frame '{frame}' is not available") from None


@register_frame
def eye(*args, **kwargs):
    return np.eye(3, dtype=np.float64)


@register_frame
def null(*args, **kwargs):
    for arg in args:
        if hasattr(arg, "frame_rotation"):
            #return np.transpose(arg.frame_rotation)
            # or equivalently
            arg.frame_rotation = np.eye(3, dtype=np.float64)
            return np.eye(3, dtype=np.float64)


def axes_perm(perm):
    """Returns rotation matrix of axes permutation"""
    ind = [("x","y","z").index(s) for s in list(perm.lower())]
    rotmat = np.zeros((3,3), dtype=np.float64)
    for i in range(3):
        rotmat[i,ind[i]] = 1.0
    return rotmat


@register_frame
def xyz(*args, **kwargs):
    return axes_perm('xyz')


@register_frame
def xzy(*args, **kwargs):
    return axes_perm('xzy')


@register_frame
def yxz(*args, **kwargs):
    return axes_perm('yxz')


@register_frame
def yzx(*args, **kwargs):
    return axes_perm('yzx')


@register_frame
def zxy(*args, **kwargs):
    return axes_perm('zxy')


@register_frame
def zyx(*args, **kwargs):
    return axes_perm('zyx')


@register_frame
def rotmat_from_tens(name, tens, op='I'):
    """Returns rotation matrix obtained by op(tens)"""
    try:
        x = np.array(tens)
        ndim = x.ndim
        shape = x.shape
    except AttributeError:
        raise AttributeError(f"'{name}' is not a tensor") from None
    if ndim != 2:
        raise ValueError(f"tensor '{name}' has inappropriate rank '{ndim}' != 2") from None
    if not all(dim == 3 for dim in shape):
        raise ValueError(f"tensor '{name}' has inappropriate shape '{shape}' != {[3]*ndim}") from None
    if np.all(np.abs(x) < _small):
        raise ValueError(f"all elements of tensor '{name}' are zero") from None
    if np.any(np.abs(x) > _large):
        raise ValueError(f"some of elements of tensor '{name}' are too large") from None
    if np.any(np.isnan(x)):
        raise ValueError(f"some of elements of tensor '{name}' are NaN") from None
    if op in _opers:
        return _opers[op](x)
    else:
        raise TypeError(f"operator '{op}' is not available")


@register_oper
def diag(mat):
    if np.any(np.abs(mat - mat.T) > _symm_tol):
        raise ValueError(f"tensor is not symmetric") from None
    try:
        diag, rotmat = np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        raise Exception("eigenvalues did not converge") from None
    return np.transpose(rotmat)


@register_oper
def tr(mat):
    return np.transpose(mat)


@register_oper
def I(mat):
    # default operator for rotmat_from_tens
    return mat
