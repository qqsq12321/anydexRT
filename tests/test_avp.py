import numpy as np
import pytest

from anydexrt.avp import extract_fingertips, VisionProInput


def _hand():
    x = np.repeat(np.eye(4, dtype=np.float32)[None], 25, axis=0)
    for i in range(25):
        x[i, :3, 3] = (i, i + .1, i + .2)
    return x


def test_extract_avp_tip_positions():
    got = extract_fingertips(_hand(), "wuji_hand")
    assert got.shape == (5, 3)
    assert np.allclose(got[:, 0], [4, 9, 14, 19, 24])


def test_extract_accepts_dict_and_object():
    class Frame:
        pass
    frame = Frame(); frame.right = _hand()
    class Stream:
        def __init__(self): self.latest = {"right": _hand()}
    assert np.allclose(extract_fingertips({"fingers": _hand()}), extract_fingertips(_hand()))
    inp = VisionProInput("unused", streamer=Stream())
    assert inp.poll()
    assert np.allclose(inp.fingertips(), extract_fingertips(_hand()))


def test_bad_shape_is_rejected():
    with pytest.raises(ValueError):
        extract_fingertips(np.zeros((21, 3)))
