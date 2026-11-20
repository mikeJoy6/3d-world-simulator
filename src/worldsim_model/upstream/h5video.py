"""Lossless video storage for image stacks inside HDF5.

A (T, H, W, 3) uint8 colour stack, a (T, H, W) uint8 infrared stack or a (T, H, W) float16 depth / disparity stack is stored as
one FFV1 packet per frame in a variable-length byte dataset of the same name. Every frame is a key frame, so a frame is read on its
own, and the decoded array is bit-exact (float16 is stored through its 16 bits, never rounded). Colour and infrared shrink to about a
third of LZF, float16 depth and disparity to about half of gzip.

    f = h5video.File(path, "r")                       # drop-in for h5py.File; plain datasets pass through untouched
    rgb = f["observations/images/camera_0_color"][t]  # decoded frame, same shape and dtype as before
    h5video.write(group, "camera_0_color", array)     # encode

`python scripts/h5_to_video.py` converts existing files.
"""
from fractions import Fraction

import h5py
import numpy as np

ATTR = "h5video"
THREADS = 4
_VLEN = h5py.vlen_dtype(np.uint8)


def _layout(shape, dtype):
    """-> (frame format given to the encoder, pixel format stored) for an array, or None when it is not stored as video."""
    dtype = np.dtype(dtype)
    if len(shape) == 4 and shape[3] == 3 and dtype == np.uint8:
        return "rgb24", "bgr0"
    if len(shape) == 3 and dtype == np.uint8:
        return "gray", "gray"
    if len(shape) == 3 and dtype in (np.float16, np.uint16):
        return "gray16le", "gray16le"
    return None


def suitable(ds, min_side=64):
    """True for a dataset worth storing as video: an image stack of at least ``min_side`` pixels a side."""
    return _layout(ds.shape, ds.dtype) is not None and ds.shape[0] > 0 and min(ds.shape[1:3]) >= min_side


def _context(mode, w, h, pix):
    import av

    c = av.CodecContext.create("ffv1", mode)
    c.width, c.height, c.pix_fmt = w, h, pix
    c.thread_type, c.thread_count = "SLICE", THREADS
    if mode == "w":
        c.time_base = Fraction(1, 30)
        c.options = {"level": "3", "g": "1", "slices": "16", "slicecrc": "0", "coder": "1", "context": "1"}
    return c


def write(group, name, frames, shape=None, dtype=None):
    """Encode ``frames`` (an array, an h5py dataset or any sequence of frames) into ``group[name]``."""
    import av

    shape = tuple(frames.shape if shape is None else shape)
    dtype = np.dtype(frames.dtype if dtype is None else dtype)
    fmt, pix = _layout(shape, dtype)
    enc = _context("w", shape[2], shape[1], pix)
    enc.open()
    out = group.create_dataset(name, shape=(shape[0],), dtype=_VLEN)
    n = 0
    for t in range(shape[0]):
        a = np.ascontiguousarray(frames[t])
        frame = av.VideoFrame.from_ndarray(a.view(np.uint16) if fmt == "gray16le" else a, format=fmt)
        frame.pts = t
        for p in enc.encode(frame if fmt == pix else frame.reformat(format=pix)):
            out[n] = np.frombuffer(bytes(p), np.uint8)
            n += 1
    for p in enc.encode(None):
        out[n] = np.frombuffer(bytes(p), np.uint8)
        n += 1
    assert n == shape[0], f"{name}: {n} packets for {shape[0]} frames"
    out.attrs[ATTR] = "ffv1"
    out.attrs["shape"], out.attrs["dtype"] = np.asarray(shape), str(dtype)
    out.attrs["extradata"] = np.frombuffer(bytes(enc.extradata), np.uint8)
    return out


class VideoDataset:
    """Reads like the h5py dataset it replaced: ``shape``, ``dtype``, ``len`` and indexing by frame, slice or list of frames,
    optionally followed by an index into the frame (``ds[t, 100:200]``)."""

    def __init__(self, ds):
        self._ds = ds
        self.shape, self.dtype = tuple(int(v) for v in ds.attrs["shape"]), np.dtype(ds.attrs["dtype"])
        self.ndim, self.name, self.attrs = len(self.shape), ds.name, ds.attrs
        self._fmt, self._pix = _layout(self.shape, self.dtype)
        self._dec = None

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    def _frame(self, t):
        import av

        if self._dec is None:
            self._dec = _context("r", self.shape[2], self.shape[1], self._pix)
            self._dec.extradata = bytes(self._ds.attrs["extradata"])
            self._dec.open()
        t = int(t) + (self.shape[0] if t < 0 else 0)
        frames = self._dec.decode(av.Packet(self._ds[t].tobytes()))
        a = frames[0].to_ndarray(format=self._fmt)
        return a.view(self.dtype) if self._fmt == "gray16le" else a

    def __getitem__(self, key):
        key, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        rest = tuple(rest)
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, (int, np.integer)):
            a = self._frame(key)
            return a[rest] if rest else a
        idx = range(*key.indices(self.shape[0])) if isinstance(key, slice) else [int(i) for i in np.asarray(key).reshape(-1)]
        out = np.empty((len(idx),) + self.shape[1:], self.dtype)
        for i, t in enumerate(idx):
            out[i] = self._frame(t)
        return out[(slice(None),) + rest] if rest else out

    def __array__(self, dtype=None, copy=None):
        a = self[:]
        return a if dtype is None else a.astype(dtype)

    def astype(self, dtype):
        return self[:].astype(dtype)


def _wrap(o):
    if isinstance(o, h5py.Dataset):
        return VideoDataset(o) if ATTR in o.attrs else o
    return Group(o) if isinstance(o, h5py.Group) else o


class Group:
    """An h5py group (or file) whose video datasets come back as :class:`VideoDataset`; everything else is h5py's own."""

    def __init__(self, g):
        self._g = g

    def __getattr__(self, k):
        return getattr(self._g, k)

    def __getitem__(self, k):
        return _wrap(self._g[k])

    def __setitem__(self, k, v):
        self._g[k] = v

    def __delitem__(self, k):
        del self._g[k]

    def __contains__(self, k):
        return k in self._g

    def __iter__(self):
        return iter(self._g)

    def __len__(self):
        return len(self._g)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._g.close()

    def get(self, k, default=None):
        return _wrap(self._g[k]) if k in self._g else default

    def values(self):
        return [_wrap(v) for v in self._g.values()]

    def items(self):
        return [(k, _wrap(v)) for k, v in self._g.items()]

    def create_group(self, *a, **kw):
        return Group(self._g.create_group(*a, **kw))

    def require_group(self, *a, **kw):
        return Group(self._g.require_group(*a, **kw))


def File(*a, **kw):
    """h5py.File, with video datasets decoded on read. Works in every mode; writing goes straight to h5py."""
    return Group(h5py.File(*a, **kw))
