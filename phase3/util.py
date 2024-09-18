from typing import TypeAlias

from phase3 import satelliteClass

CommandersIndent: TypeAlias = dict[int, dict[satelliteClass.Satellite, dict[int, int]]]


class NestedDict(dict):
    def __missing__(self, key):
        value = self[key] = NestedDict()
        return value

    def __call__(self):
        return 0
