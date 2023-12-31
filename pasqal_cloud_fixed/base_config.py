from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, ForwardRef, List, Optional

from .result_type import MyResultType


class InvalidConfiguration(Exception):
    pass


INVALID_KEY_ERROR_MSG = (
    "Invalid key {} in Configuration.extra_config. "
    + "Attempted to override a default field."
)
INVALID_RESULT_TYPES = (
    "Invalid set result types {} in Configuration. This device allows for {}."
)


@dataclass
class MyBaseConfig:
    """Base class for all device configuration classes.

    This class provides a common interface for all device configuration classes.
    It also provides a mechanism to add extra configuration parameters
    that are not part of the default configuration.

    The extra configuration parameters are stored in the `extra_config` field.
    This field is a dictionary that can be used to store any extra configuration
    parameters. The `extra_config` field is not part of the configuration
    dictionary returned by the `to_dict` method. Instead, the extra configuration
    parameters are added to the configuration dictionary.
    """

    # temp = MyResultType()

    extra_config: Optional[Dict[str, Any]] = None
    result_types: Optional[List[MyResultType]] = None

    def to_dict(self) -> dict[str, Any]:
        """Converts the configuration to a dictionary.

        The extra configuration parameters are added to the configuration
        dictionary.

        Returns:
                                                                        dict: Configuration dictionary.
        """
        self._validate()
        return self._unnest_extra_config(asdict(self))

    @classmethod
    def from_dict(cls, conf: dict[str, Any]) -> MyBaseConfig:
        """Creates a configuration object from a dictionary.

        The extra configuration parameters are stored in the `extra_config` field.

        Args:
                                                                        conf (dict): Configuration dictionary.

        Returns:
                                                                        BaseConfig: Configuration object.
        """
        base_conf = {}
        for field in fields(cls):
            if field.name != "extra_config" and field.name in conf:
                base_conf[field.name] = conf.pop(field.name)

        # ensure that no extra config is passed as None
        if not conf:
            conf = None  # type: ignore
        return cls(**base_conf, extra_config=conf)

    @staticmethod
    def _unnest_extra_config(conf_dict: dict[str, Any]) -> dict[str, Any]:
        res = {k: v for (k, v) in conf_dict.items() if not k == "extra_config"}
        if conf_dict.get("extra_config", None):
            res.update(conf_dict["extra_config"])
        return res

    @property
    def allowed_result_types(self) -> List[str]:
        return ["counter"]

    def _validate(self) -> None:
        if self.extra_config:
            for k in self.extra_config.keys():
                if k in self.__dataclass_fields__.keys():
                    raise InvalidConfiguration(INVALID_KEY_ERROR_MSG.format(k))
        if self.result_types:
            if not set(self.result_types) <= set(self.allowed_result_types):
                raise InvalidConfiguration(
                    INVALID_RESULT_TYPES.format(
                        self.result_types, self.allowed_result_types
                    )
                )
