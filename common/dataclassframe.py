import dataclasses
from typing import ClassVar, Generic, Protocol, Type, TypeVar

import pandas as pd


class DataclassLike(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]


T = TypeVar('T', bound=DataclassLike)


class DataClassFrame(pd.DataFrame, Generic[T]):
    """A DataFrame with wrappers to insert and extract dataclasses."""

    # Pandas overrides so we can set `self._clz`
    _internal_names = pd.DataFrame._internal_names + ['_clz']  # type: ignore
    _internal_names_set = set(_internal_names)

    def __init__(self, data: list[T] | None = None, *args, clz: Type[T], **kwargs):
        self._clz = clz
        if 'columns' in kwargs:
            raise ValueError('columns must not be provided')
        data_dicts = [dataclasses.asdict(d) for d in data] if data else None
        kwargs['columns'] = [f.name for f in dataclasses.fields(clz)]
        super().__init__(data_dicts, *args, **kwargs)  # type: ignore

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def append(self, data: T) -> None:
        self.loc[len(self)] = dataclasses.asdict(data)  # type: ignore

    def to_dataclasses(self, other: pd.DataFrame | None = None) -> list[T]:
        if other is None:
            other = self
        return other.apply(lambda row: self._clz(*row), axis=1).tolist()  # type: ignore
