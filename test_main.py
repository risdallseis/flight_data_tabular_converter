import pytest
import numpy as np
from ..main import up_sample_array

@pytest.mark.parametrize(
    'array, rate, result',
    [
        (
            np.array([1, 2, 3, 4]),
            1,
            np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.float32)
            )

    ]
)
def test_up_sample_array(array, rate, result):
    """unit test for upsample array"""
    np.testing.assert_array_equal(up_sample_array(array, rate, max_rate=4), result)