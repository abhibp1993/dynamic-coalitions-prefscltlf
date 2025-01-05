class State:
    def __init__(self, **kwargs):
        # Internal representation
        self._var_names = list(sorted(kwargs.keys()))
        self._values = [kwargs[name] for name in self._var_names]

    def __repr__(self):
        return f"State({', '.join(f'{self._var_names[i]}:{self._values[i]}' for i in range(len(self._var_names)))})"

    def __str__(self):
        return f"State({', '.join(f'{self._var_names[i]}:{self._values[i]}' for i in range(len(self._var_names)))})"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other: 'State'):
        """
        Given that variable names are sorted, two states are equal if and only if their var_names are equal and
        their (tuple) representations are equal.
        """
        return self._var_names == other._var_names and self._values == other._values

    def __getattribute__(self, item):
        try:
            var_names = object.__getattribute__(self, "_var_names")
            if item in var_names:
                idx = var_names.index(item)
                return self._values[idx]
            return object.__getattribute__(self, item)

        except AttributeError:
            return object.__getattribute__(self, item)

    def __getstate__(self):
        return {self._var_names[i]: self._values[i] for i in range(len(self._var_names))}

    def __setstate__(self, state):
        self._var_names = list(sorted(state.keys()))
        self._values = [state[name] for name in self._var_names]

    def vars(self):
        return self._var_names

    def pretty(self):
        st = (f"{name}:{val}" for name, val in zip(self._var_names, self._values))
        return "State(" + ", ".join(st) + ")"
