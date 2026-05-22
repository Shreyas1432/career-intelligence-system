import types
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel


def create_mock_pydantic[T: BaseModel](model_class: type[T], **overrides: Any) -> T:
    """
    Dynamically creates an instance of a Pydantic model with mock data.
    Inspects fields, resolves standard types, and overrides with provided values.
    """
    data = {}
    for name, field in model_class.model_fields.items():
        if name in overrides:
            data[name] = overrides[name]
            continue

        # Check if default is defined (PydanticUndefined has str representation "PydanticUndefined")
        if field.default is not None and str(field.default) != "PydanticUndefined":
            data[name] = field.default
            continue

        if field.default_factory is not None:
            data[name] = field.default_factory()  # type: ignore[call-arg]
            continue

        # Generate value based on annotation type
        data[name] = _generate_mock_value(field.annotation, name)

    return model_class.model_validate(data)


def _generate_mock_value(annotation: Any, name: str) -> Any:
    """
    Recursively generates mock values based on types.
    """
    if annotation is None or annotation is type(None):
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Union types (e.g. Optional[str] -> Union[str, None])
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        return _generate_mock_value(non_none_args[0], name) if non_none_args else None

    # Handle standard collections
    if origin is list:
        return [_generate_mock_value(args[0] if args else str, name)]

    if origin is dict:
        key_t = args[0] if args else str
        val_t = args[1] if len(args) > 1 else str
        return {_generate_mock_value(key_t, name): _generate_mock_value(val_t, name)}

    if origin is set:
        return {_generate_mock_value(args[0] if args else str, name)}

    # Handle nested BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return create_mock_pydantic(annotation)

    # Basic types mapping
    basic_types = {
        str: f"mock_{name}",
        int: 1,
        float: 1.0,
        bool: True,
    }
    if annotation in basic_types:
        return basic_types[annotation]

    # Fallback for other annotations
    return f"mock_{name}"
