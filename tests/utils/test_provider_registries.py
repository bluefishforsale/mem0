"""Vector store providers are registered twice, in two files, keyed the same way.

`VectorStoreFactory.provider_to_class` maps a provider name to its implementation.
`VectorStoreConfig._provider_configs` maps the same name to its config class, as a
*string* resolved by dynamic import at validation time.

Nothing connects them. Adding a store to one and not the other fails only at
runtime, and differently depending on which was missed: omit the factory entry
and config validation passes before `create` raises; omit the config entry and
validation raises "Unsupported vector store provider" while the factory knows
the provider perfectly well. Renaming a config class breaks nothing until a
caller happens to pick that provider.

These tests are the connection.
"""

import importlib

import pytest

from mem0.utils.factory import VectorStoreFactory
from mem0.vector_stores.configs import VectorStoreConfig

# A pydantic private attribute, so the mapping lives on the descriptor's default
# rather than on the class itself.
CONFIG_MAP = VectorStoreConfig.__private_attributes__["_provider_configs"].default


def test_every_factory_provider_has_a_config():
    missing = sorted(set(VectorStoreFactory.provider_to_class) - set(CONFIG_MAP))
    assert not missing, (
        f"registered in VectorStoreFactory but not VectorStoreConfig: {missing}. "
        "Config validation will reject these as unsupported providers."
    )


def test_every_config_provider_has_a_factory_entry():
    missing = sorted(set(CONFIG_MAP) - set(VectorStoreFactory.provider_to_class))
    assert not missing, (
        f"registered in VectorStoreConfig but not VectorStoreFactory: {missing}. "
        "Config validation will pass and then create() will raise."
    )


@pytest.mark.parametrize("provider", sorted(CONFIG_MAP))
def test_config_class_name_resolves(provider):
    """The config side stores a class *name*, imported dynamically, so a rename
    in mem0/configs/vector_stores/ breaks nothing until someone selects that
    provider. This resolves every name the way validate_and_create_config does.
    """
    class_name = CONFIG_MAP[provider]
    try:
        module = importlib.import_module(f"mem0.configs.vector_stores.{provider}")
    except ImportError as e:
        # Several config modules import their provider SDK at module scope. A
        # missing optional dependency is an environment fact, not a registry bug.
        pytest.skip(f"{provider} config needs an optional dependency: {e}")

    assert hasattr(module, class_name), (
        f"VectorStoreConfig maps {provider!r} to {class_name!r}, which does not exist "
        f"in mem0.configs.vector_stores.{provider}"
    )
