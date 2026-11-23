"""Fetch only the files we need from the 12.6 GB ZebraMap_DB.zip on Zenodo (CC BY 4.0).

Zenodo supports HTTP range requests, so the zip is read remotely: the central directory
(the zip's table of contents, at the end of the file) is downloaded first, then each wanted
member is downloaded with many parallel range requests and decompressed locally.
Images are never downloaded.

    python scripts/zebramap_fetch.py --list
    python scripts/zebramap_fetch.py --get structured_cases.json --threads 24
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "https://zenodo.org/records/17623607/files/ZebraMap_DB.zip?download=1"
OUT = Path(__file__).resolve().parent.parent / "data" / "zebramap"
_local = threading.local()


def sess() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def get_range(start: int, end: int, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            r = sess().get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"range {start}-{end} failed")


def size() -> int:
    r = requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=60)
    return int(r.headers["Content-Range"].split("/")[1])


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of the remote zip (for reading the central directory)."""

    def __init__(self, total: int):
        self.total, self.pos, self.cache = total, 0, {}

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.total - self.pos
        n = min(n, self.total - self.pos)
        if n <= 0:
            return b""
        data = get_range(self.pos, self.pos + n - 1)
        self.pos += n
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def parallel_download(start: int, length: int, threads: int, chunk: int = 8 << 20) -> bytes:
    parts = [(o, min(o + chunk, start + length) - 1) for o in range(start, start + length, chunk)]
    done, lock, t0 = [0], threading.Lock(), time.time()
    buf = bytearray(length)

    def job(p):
        a, b = p
        data = get_range(a, b)
        buf[a - start:a - start + len(data)] = data
        with lock:
            done[0] += len(data)
            mb = done[0] / 1e6
            sys.stdout.write(f"\r  {mb:,.0f} / {length / 1e6:,.0f} MB  {mb / max(time.time() - t0, .1):.1f} MB/s   ")
            sys.stdout.flush()

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(job, parts))
    print()
    return bytes(buf)


def fetch_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, threads: int) -> Path:
    # local header: 30 bytes + name + extra; its extra length may differ from the central one
    hdr = get_range(info.header_offset, info.header_offset + 29)
    n_name, n_extra = struct.unpack("<HH", hdr[26:30])
    data_start = info.header_offset + 30 + n_name + n_extra
    raw = parallel_download(data_start, info.compress_size, threads)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(raw) & 0xFFFFFFFF != info.CRC:
        raise RuntimeError("CRC mismatch")
    out = OUT / Path(info.filename).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    total = size()
    zf = zipfile.ZipFile(io.BufferedReader(RemoteFile(total), buffer_size=1 << 20))
    infos = zf.infolist()
    if a.list:
        non_img = [i for i in infos if not i.filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"))]
        print(f"{len(infos):,} entries, {len(infos) - len(non_img):,} images; other files:")
        for i in non_img:
            print(f"  {i.filename}  {i.compress_size / 1e6:.1f} MB compressed, {i.file_size / 1e6:.1f} MB")
    for name in a.get:
        info = next((i for i in infos if i.filename.endswith(name)), None)
        if not info:
            print("not found:", name)
            continue
        t0 = time.time()
        print(f"{info.filename}: {info.compress_size / 1e6:.1f} MB compressed")
        out = fetch_member(zf, info, a.threads)
        print(f"  saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
