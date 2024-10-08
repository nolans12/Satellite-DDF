import pathlib

from prompt_toolkit import document
from prompt_toolkit import validation


class PathValidator(validation.Validator):
    def __init__(
        self,
        message: str,
        is_file: bool = False,
        is_dir: bool = False,
        must_exist: bool = True,
    ):
        self._message = message
        self._is_file = is_file
        self._is_dir = is_dir
        self._must_exist = must_exist

    def validate(self, document: document.Document) -> None:
        """Check if user input is a filepath that exists on the system based on conditions.

        This method is used internally by `prompt_toolkit <https://python-prompt-toolkit.readthedocs.io/en/master/>`_.

        See Also:
            https://python-prompt-toolkit.readthedocs.io/en/master/pages/asking_for_input.html?highlight=validator#input-validation
        """
        path = pathlib.Path(document.text).expanduser()

        if not path.exists():
            if self._must_exist:
                raise validation.ValidationError(
                    message=self._message,
                    cursor_position=document.cursor_position,
                )
            return

        if self._is_file and not path.is_file():
            raise validation.ValidationError(
                message=self._message,
                cursor_position=document.cursor_position,
            )
        elif self._is_dir and not path.is_dir():
            raise validation.ValidationError(
                message=self._message,
                cursor_position=document.cursor_position,
            )
