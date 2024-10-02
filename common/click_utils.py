"""Click utilities for the project

Borrowed vertabim from Ryan's personal repo
"""

import enum
from typing import Any, Type

import rich_click as click


class EnumChoice(click.Choice):
    """Use an Enum as choices for a click option

    Borrowed from https://github.com/pallets/click/pull/2210
    """

    def __init__(self, enum_type: Type[enum.Enum], case_sensitive: bool = False):
        super().__init__(
            choices=[element.name for element in enum_type],
            case_sensitive=case_sensitive,
        )
        self.enum_type = enum_type

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> Any:
        if not isinstance(value, str):
            # Assume enum instance
            value = value.value
        value = super().convert(value=value, param=param, ctx=ctx)
        if value is None:
            return None
        return self.enum_type[value]
