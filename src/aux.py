import json
import os
import types
import warnings


class ConfigError(Exception):
    pass


def config_from_dict(d, validate=True):
    config = types.SimpleNamespace(**d)

    if validate:
        validate_config(config)

    return config


def load_config(path, validate=True):
    with open(path) as fd:
        return config_from_dict(json.load(fd), validate=validate)


def apply_overrides(config, **overrides):
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ConfigError(f"Unknown config key: '{key}'")
        original_type = type(getattr(config, key))
        try:
            if original_type is bool:
                if value.lower() in ("true", "1", "yes"):
                    setattr(config, key, True)
                elif value.lower() in ("false", "0", "no"):
                    setattr(config, key, False)
                else:
                    raise ConfigError(f"Cannot cast '{value}' to bool for key '{key}'")
            else:
                setattr(config, key, original_type(value))
        except (ValueError, TypeError) as e:
            raise ConfigError(
                f"Cannot cast '{value}' to {original_type.__name__} for key '{key}': {e}"
            )
    return config


def validate_config(config):
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ConfigError(
            f"num_attention_heads ({config.num_attention_heads}) must be "
            f"divisible by num_key_value_heads ({config.num_key_value_heads})"
        )

    head_dim = config.hidden_size // config.num_attention_heads
    if head_dim != 64:
        warnings.warn(f"head_dim = {head_dim} (typically 64)")

    ratio = config.intermediate_size / config.hidden_size
    if ratio != 4.0:
        warnings.warn(f"intermediate_size / hidden_size = {ratio:.1f} (typically 4)")

    for attr in ["tokenizer_path", "train_data", "val_data"]:
        path = getattr(config, attr)
        if not os.path.exists(path):
            raise ConfigError(f"File not found: {attr} = '{path}'")
