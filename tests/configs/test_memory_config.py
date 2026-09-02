"""What MemoryConfig must refuse.

Pydantic's default is ``extra="ignore"``, so for most of this config's life a
key it did not define was dropped without a word. That makes every option
weaker than it looks: a misspelt knob reads as accepted and does nothing, and
so does a block the SDK used to honour and no longer does.
"""

import pytest
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig


def test_a_removed_config_block_is_rejected_not_dropped():
    """`graph_store` was removed from the OSS SDK in v3, and
    docs/migration/oss-v2-to-v3.mdx tells people to delete the block because it
    "is no longer read". Anyone who skipped that step got silence: the config
    was accepted and graph memory simply never happened.
    """
    with pytest.raises(ValidationError, match="graph_store"):
        MemoryConfig(graph_store={"provider": "neo4j"})
