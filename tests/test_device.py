from deoldify.device_id import DeviceId
from deoldify._device import _Device


def test_device_default_cpu():
    d = _Device()
    assert d.current() == DeviceId.CPU
    assert not d.is_gpu()


def test_device_set_cpu():
    d = _Device()
    d.set(DeviceId.CPU)
    assert d.current() == DeviceId.CPU
    assert not d.is_gpu()


def test_device_set_gpu():
    d = _Device()
    d.set(DeviceId.GPU0)
    assert d.current() == DeviceId.GPU0
    assert d.is_gpu()
    # Reset to CPU
    d.set(DeviceId.CPU)
