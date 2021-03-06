"""Specifications declare the expected variables layout of CTranslate2 models
that do not load a computation graph. The model converter should make sure that
each required variable of the specification is set.
"""

import struct
import six

OPTIONAL = "optional"


def _join_scope(scope, name):
    if not scope:
        return name
    return "%s/%s" % (scope, name)

def _split_scope(scope):
    return scope.split("/")

def visit_spec(spec, fn, scope=""):
    """Recursively visits a layer spec."""
    for name, value in list(six.iteritems(spec.__dict__)):
        if isinstance(value, list):
            for i, elem in enumerate(value):
                visit_spec(elem, fn, scope=_join_scope(scope, "%s_%d" % (name, i)))
        elif isinstance(value, LayerSpec):
            visit_spec(value, fn, scope=_join_scope(scope, name))
        else:
            fn(spec, _join_scope(scope, name), value)


class LayerSpec(object):
    """Layer specification."""

    @property
    def revision(self):
        return 1

    def validate(self):
        """Checks that required variables are set to a valid value."""
        import numpy as np
        def _check(spec, name, value):
            if value is None:
                raise ValueError("Missing value for attribute %s" % name)
            # Promote float16 to float32 as it is currently an unsupported type.
            if isinstance(value, np.ndarray) and value.dtype == np.float16:
                attr_name = _split_scope(name)[-1]
                setattr(spec, attr_name, value.astype(np.float32))
        self.visit(_check)

    def variables(self, prefix=""):
        """Returns a dict mapping variables name to value."""
        var = {}
        def _register_var(spec, name, value):
            if isinstance(value, six.string_types) and value == OPTIONAL:
                return
            var[_join_scope(prefix, name)] = value
        self.visit(_register_var)
        return var

    def quantize(self, quantization):
        """Possibly quantizes the variable of the layer."""
        import numpy as np
        def _quantize(spec, name, value):
            if "weight" in name:
                if quantization == "int16":
                    # Represent the value with 10 bits so the multiplication is 20 bits
                    # and 12 bits are left for accumulation.
                    scale = np.dtype(value.dtype).type(2**10 / np.amax(np.absolute(value)))
                    value *= scale
                    value = np.clip(value, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
                    value = value.astype(np.int16)
                elif quantization == "int8":
                    scale = 127.0 / np.amax(np.absolute(value), axis=1)
                    value *= np.expand_dims(scale, 1)
                    value = value.astype(np.int8)
                setattr(spec, "weight_scale", scale)
                setattr(spec, "weight", value)
        self.visit(_quantize)

    def visit(self, fn):
        """Recursively visits this layer and its children."""
        visit_spec(self, fn)

    def serialize(self, path):
        """Serializes this specification."""
        variables = self.variables()
        with open(path, "wb") as model:

            def _write_string(string):
                model.write(struct.pack("H", len(string) + 1))
                model.write(six.b(string))
                model.write(struct.pack('B', 0))

            model.write(struct.pack("I", 2))  # Binary version.
            _write_string(self.__class__.__name__)
            model.write(struct.pack("I", self.revision))
            model.write(struct.pack("I", len(variables)))
            for name, value in sorted(six.iteritems(variables), key=lambda x: x[0]):
                _write_string(name)
                model.write(struct.pack("B", len(value.shape)))
                for dim in value.shape:
                    model.write(struct.pack("I", dim))
                model.write(struct.pack("B", value.dtype.itemsize))
                model.write(struct.pack("I", value.size))
                model.write(value.tobytes())
