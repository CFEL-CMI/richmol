import numpy as np
import mol_frames


_small = abs(np.finfo(float).eps)
_large = abs(np.finfo(float).max)

_tensors = dict()

def register_tensor(arg):
    if callable(arg):
        raise ValueError(f"please specify rank in @register_tensor(<rank>) for function '{arg.__name__}'") from None
    else:
        rank = arg
    def _decorator(func):
        _tensors[func.__name__] = [rank, func, None] # [rank, (rotation) function, ref values]
        return func
    return _decorator


def dynamic_rotation(obj, attr_name, attr_value):
    if attr_name not in _tensors:
        return
    # check tensor values
    try:
        x = np.array(attr_value)
        ndim = x.ndim
        shape = x.shape
    except AttributeError:
        raise AttributeError(f"'{attr_name}' is not a tensor") from None
    rank = _tensors[attr_name][0]
    if ndim != rank:
        raise ValueError(f"tensor '{attr_name}' has inappropriate rank '{ndim}' != {rank}") from None
    if not all(dim == 3 for dim in shape):
        raise ValueError(f"tensor '{attr_name}' has inappropriate shape '{shape}' != {[3]*ndim}") from None
    if np.all(np.abs(x) < _small):
        raise ValueError(f"all elements of tensor '{attr_name}' are zero") from None
    if np.any(np.abs(x) > _large):
        raise ValueError(f"some of elements of tensor '{attr_name}' are too large") from None
    if np.any(np.isnan(x)):
        raise ValueError(f"some of elements of tensor '{attr_name}' are NaN") from None
    # register tensor values
    _tensors[attr_name][2] = x
    # function
    func = _tensors[attr_name][1]
    # add property to object, keep the original tensor values
    # setattr(obj, attr_name + "_ref", attr_value)
    setattr(obj, attr_name, property(func)) # obj.attr_name will now call a function (defined below)


@register_tensor(1)
def dip(self):
    """Frame rotation of dipole moment"""
    return np.dot(self.frame, _tensors['dip'][2])


@register_tensor(2)
def pol(self):
    """Frame rotation of polarizability"""
    return np.dot(self.frame, np.dot(_tensors['pol'][2], np.transpose(self.frame)))


@register_tensor(2)
def inertia(self):
    """Frame rotation of moment of inertia tensor"""
    itens = self.imom()
    # don't perform rotation of tensor because it is computed from Cartesian
    # coordinates of atoms, which are automatically rotated
    return itens


@register_tensor(2)
def ipas(self):
    """Matrix rotating inertia tensor to diagonal form"""
    return mol_frames.diag(self.imom())
