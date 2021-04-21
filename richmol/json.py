"""Custom encoders and decoders for JSON"""
import numpy as np
import types
import json
from collections.abc import Mapping


_encoders = dict()
_decoders = dict()


def register_encoder(arg):
    if isinstance(arg, str):
        arg_type = arg
    else:
        arg_type = arg.__class__.__name__
    def _decorator(func):
        _encoders[arg_type] = func
        return func
    return _decorator


def register_decoder(arg):
    try:
        if arg.startswith("__"):
            key = arg
        else:
            raise AttributeError
    except AttributeError:
        raise AttributeError("illegal decorator parameter '{arg}'") from None
    def _decorator(func):
        _decoders[key] = func
        return func
    return _decorator


class Encoder(json.JSONEncoder):
    def default(self, obj):
        obj_type = obj.__class__.__name__
        if obj_type in _encoders:
            return _encoders[obj_type](obj)
        else:
            return super().default(obj)


def decode(dct):
    for dec in _decoders.keys():
        if dec in dct.keys():
            return _decoders[dec](dct)
    return dct


def loads(var):
    jl = json.loads(var, object_hook=decode)
    if isinstance(jl, Mapping):
        jl = parse_keys(jl)
    return jl


def dumps(var):
    if isinstance(var, Mapping):
        var_ = parse_keys_rev(var)
    else:
        var_ = var
    jd = json.dumps(var_, cls=Encoder, sort_keys=True)
    return jd


def parse_keys(node):
    """Converts keys in nested dictionary from strings to integers and floats where possible
    Rounds floats to first decimal
    """
    out = dict()
    for key, item in node.items():
        if isinstance(item, Mapping):
            item_ = parse_keys(item)
        else:
            item_ = item
        try:
            if key.find('.') == -1:
                key_ = int(key)
            else:
                key_ = round(float(key), 1)
        except ValueError:
            key_ = key
        out[key_] = item_
    return out


def parse_keys_rev(node):
    """Converts keys in nested dictionary from integers and floats into strings
    Rounds floats to first decimal
    """
    out = dict()
    for key, item in node.items():
        if isinstance(item, Mapping):
            item_ = parse_keys_rev(item)
        else:
            item_ = item
        try:
            if isinstance(key, float):
                key_ = str(round(key, 1))
            else:
                key_ = int(key)
        except ValueError:
            key_ = key
        out[key_] = item_
    return out


@register_encoder(1+1j)
def encode_complex(obj):
    """Complex number"""
    return {"real":obj.real, "imag":obj.imag, "__complex__":True}


@register_decoder("__complex__")
def decode_complex(dct):
    return complex(dct["real"], dct["imag"])


@register_encoder(np.ndarray(1))
def encode_ndarray(obj):
    """Numpy ndarray"""
    res = {"val":obj.tolist(), "__ndarray__":True}
    try:
        # structured array
        dtype = []
        for x,y in obj.dtype.fields.items():
            sdt = y[0].subdtype
            if sdt is not None:
                dtype.append((x, str(sdt[0]), sdt[1]))
            else:
                dtype.append((x, str(y[0])))
        res["dtype"] = dtype
    except AttributeError:
        pass
    return res


@register_decoder("__ndarray__")
def decode_ndarray(dct):
    if "dtype" in dct:
        # structured array
        dtype = [(elem[0], elem[1], elem[2]) if len(elem)>2 else (elem[0], elem[1])
                 for elem in dct["dtype"]]
        return np.array([tuple(elem) for elem in dct["val"]], dtype=dtype)
    else:
        # normal array
        return dct["val"]


@register_encoder(type("", (object,), {}))
def encode_type(obj):
    """Type object"""
    res = {name:getattr(obj, name) for name in vars(obj).keys()}
    res["__type__"] = True
    return res


@register_decoder("__type__")
def decode_type(dct):
    return type("name", (object,), {key:val for key,val in dct.items()})


@register_encoder(lambda x:x)
def encode_function(obj):
    """FunctionType"""
    return None