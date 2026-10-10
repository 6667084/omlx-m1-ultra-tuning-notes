#!/usr/bin/env python3
"""把 PyTorch zip 格式的 .bin 检查点转成 safetensors，不依赖 torch，也不执行任意 pickle 代码。

反序列化只放行白名单全局对象（OrderedDict、torch._utils._rebuild_tensor_v2、torch.*Storage），
其余一律抛 UnpicklingError——这等价于 torch.load(weights_only=True) 的安全模型。
用法：bin2st.py <pytorch_model.bin> <输出 model.safetensors>
"""
import collections
import json
import pickle
import sys
import zipfile

import numpy as np
from safetensors.numpy import save_file

STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.name = name
        self.dtype = STORAGE_DTYPES[name]


class _Storage:
    def __init__(self, array):
        self.array = array


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    arr = storage.array
    itemsize = arr.itemsize
    view = np.lib.stride_tricks.as_strided(
        arr[offset:],
        shape=tuple(size),
        strides=tuple(s * itemsize for s in stride),
    )
    return np.ascontiguousarray(view)


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, fh, zf, prefix):
        super().__init__(fh)
        self._zf = zf
        self._prefix = prefix
        self._storages = {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in STORAGE_DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        tag, storage_type, key, _location, numel = pid
        if tag != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {tag!r}")
        if key not in self._storages:
            raw = self._zf.read(f"{self._prefix}/data/{key}")
            arr = np.frombuffer(raw, dtype=storage_type.dtype, count=numel)
            self._storages[key] = _Storage(arr)
        return self._storages[key]


def load_bin(path):
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with zf.open(pkl) as fh:
            return _SafeUnpickler(fh, zf, prefix).load()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    state = load_bin(src)
    tensors = {k: v for k, v in state.items()}
    n_params = sum(int(np.prod(v.shape)) for v in tensors.values())
    dtypes = collections.Counter(str(v.dtype) for v in tensors.values())
    print(f"tensors={len(tensors)} params={n_params:,} dtypes={dict(dtypes)}")
    for k in list(tensors)[:6] + ["..."] + list(tensors)[-6:]:
        print("  ", k, tuple(tensors[k].shape) if k != "..." else "")
    save_file(tensors, dst, metadata={"format": "pt"})
    print("saved", dst)
    json.dump(
        {k: list(v.shape) for k, v in tensors.items()},
        open(dst + ".shapes.json", "w"),
    )


if __name__ == "__main__":
    main()
