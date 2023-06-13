import logging
from xarray_data_accessor.data_converters.base import DataConverterBase
from typing import (
    Dict,
    List,
)


class DataConversionFactory:
    """A factory for data converters.

    Protected Attributes:
        __data_converters: A dictionary of names : registered data converters.
        __conversion_function_provenance: Associated conversion function names
            with the data converter that registered them.
        __conversion_functions: A dictionary of names : conversion functions.
    """

    __data_converters: Dict[str, DataConverterBase] = {}
    __conversion_function_provenance: Dict[str, DataConverterBase] = {}
    __conversion_functions: Dict[
        str,
        DataConverterBase.ConversionFunctionType,
    ] = {}

    @classmethod
    def register(
        cls,
        data_converter: DataConverterBase,
    ) -> None:
        """Registers a converter with the factory."""
        if not issubclass(data_converter, DataConverterBase):
            raise TypeError(
                f'{data_converter.__name__} must be a subclass of '
                f'DataConverterBase.',
            )
        for func in DataConverterBase.__abstractmethods__:
            if not hasattr(data_converter, func):
                raise TypeError(
                    f'Please provide a valid model implementation! '
                    f'Model is missing {func} method.',
                )

        # register the data conversion functions
        len_i = len(cls.__conversion_functions.keys())
        converter_functions = data_converter.get_conversion_functions()
        for name, func in converter_functions.items():
            if name in cls.__conversion_functions.keys():
                raise KeyError(
                    f'Function name={name} is already registered with the '
                    f'DataConversionFactory. The previously registered '
                    f'function is from class {cls.__conversion_function_provenance[name].__name__}.',
                )
            cls.__conversion_functions[name] = func
            cls.__conversion_function_provenance[name] = data_converter
            logging.info(
                f'Registered function {name} from class {data_converter.__name__}.',
            )

        if len(cls.__conversion_functions) != len_i:
            cls.__data_converters[data_converter.__name__] = data_converter

            # update the data conversion functions
            DataConversionFunctions.add_functions()

    @classmethod
    def unregister(
        cls,
        data_converter: DataConverterBase,
    ) -> None:
        """Un-registers a converter with the factory."""
        try:
            del cls.__data_converters[data_converter.__name__]
        except KeyError:
            raise KeyError(
                f'{data_converter.__name__} is not registered with the '
                f'DataConversionFactory.',
            )

    @classmethod
    def get_functions(
        cls,
    ) -> Dict[str, DataConverterBase.ConversionFunctionType]:
        """Returns a dictionary of conversion functions."""
        return cls.__conversion_functions

    @classmethod
    def get_function_provenance(
        cls,
    ) -> Dict[str, DataConverterBase]:
        """Returns a dictionary of conversion function provenance."""
        return cls.__conversion_function_provenance

    @classmethod
    def get_converter_classes(
        cls,
    ) -> Dict[str, DataConverterBase]:
        return cls.__data_converters


class DataConversionFunctions:
    """A class for accessing data conversion functions."""

    __data_conversion_factory = DataConversionFactory
    __added_functions: List[str] = []

    @classmethod
    def get_factory(cls) -> DataConversionFactory:
        """Returns the factory."""
        return cls.__data_conversion_factory

    @classmethod
    def get_function_names(cls) -> List[str]:
        """Returns a list of the added functions."""
        return cls.__added_functions

    @classmethod
    def add_functions(cls):
        # add the data conversion functions
        for name, func in cls.__data_conversion_factory.get_functions().items():
            if name not in cls.__dict__.keys():
                setattr(cls, name, func)
                cls.__added_functions.append(name)
