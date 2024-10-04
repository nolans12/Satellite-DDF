from typing import TypeAlias

# Map of sim time (minutes) to a map of satellite names to a map of intents,
# where an intent is a map of target IDs to the accuracy required to track the target.
CommandersIndent: TypeAlias = dict[int, dict[str, dict[int, int]]]


class NestedDict(dict):
    def __missing__(self, key):
        value = self[key] = NestedDict()
        return value

    def __call__(self):
        return 0
