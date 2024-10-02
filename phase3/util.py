from typing import TypeAlias

from phase3 import satellite

CommandersIndent: TypeAlias = dict[int, dict[satellite.Satellite, dict[int, int]]]


class NestedDict(dict):
    def __missing__(self, key):
        value = self[key] = NestedDict()
        return value

    def __call__(self):
        return 0
