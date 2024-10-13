from typing import TypeAlias

# Map of sim time (minutes) to a map of satellite names to a map of intents,
# where an intent is a map of target IDs to the accuracy required to track the target.
CommandersIndent: TypeAlias = dict[int, dict[str, dict[int, int]]]
