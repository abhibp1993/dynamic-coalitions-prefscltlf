from collections.abc import Hashable


class State:
    def __init__(self, obj, **kwargs):
        assert isinstance(obj, Hashable), f"State object must be hashable."
        self._obj = obj

    def __hash__(self):
        return hash(self._obj)

    def __str__(self):
        return f"State({self._obj})"

    def __repr__(self):
        return f"State({self._obj})"

    def __eq__(self, other: 'State') -> bool:
        if isinstance(other, State):
            return self._obj == other._obj
        return False

    def get_object(self):
        return self._obj


class StateTuple(State):
    def __init__(self, obj, **kwargs):
        super().__init__(obj, **kwargs)
        assert isinstance(obj, tuple), f"State object must be tuple, received {obj=} of type {type(obj)}."

    def __getitem__(self, item):
        return self._obj[item]
