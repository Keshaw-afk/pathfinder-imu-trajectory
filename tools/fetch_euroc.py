#!/usr/bin/env python3
"""
Fetch only the IMU and ground-truth CSVs of EuRoC MAV sequences.

The official archives on the ETH Research Collection are multi-GB zips that
contain one zip per sequence (images included). The server supports HTTP
range requests, so we read the zip central directories remotely and download
just ``mav0/imu0/data.csv`` and ``mav0/state_groundtruth_estimate0/data.csv``
(a few MB per sequence) instead of ~13 GB.

    python tools/fetch_euroc.py MH_01_easy MH_02_easy ... [--out data/raw]
"""

import argparse
import io
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

ARCHIVES = {
    "machine_hall": "https://www.research-collection.ethz.ch/server/api/core/"
                    "bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content",
    "vicon_room1": "https://www.research-collection.ethz.ch/server/api/core/"
                   "bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content",
    "vicon_room2": "https://www.research-collection.ethz.ch/server/api/core/"
                   "bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content",
}
WANTED = ("mav0/imu0/data.csv", "mav0/state_groundtruth_estimate0/data.csv")
UA = "pathfinder-dataset-fetch/1.0"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of [base, base+size) of a remote file."""

    def __init__(self, url, base=0, size=None, block=1 << 20, delay=1.0):
        self.url, self.base, self.block, self.delay = url, base, block, delay
        if size is None:
            size = self._total() - base
        self.size, self.pos = size, 0
        self.cache = {}

    def _get(self, start, end):
        time.sleep(self.delay)  # be polite to the server
        req = urllib.request.Request(self.url, headers={
            "Range": "bytes=%d-%d" % (start, end), "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def _total(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Range"].split("/")[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b = self.pos // self.block
            if b not in self.cache:
                s = self.base + b * self.block
                e = min(self.base + self.size, s + self.block) - 1
                if len(self.cache) > 16:
                    self.cache.clear()
                self.cache[b] = self._get(s, e)
            data = self.cache[b]
            o = self.pos - b * self.block
            chunk = data[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def member_data_offset(f, info):
    """Absolute offset of a member's (compressed) data inside archive f."""
    f.seek(info.header_offset)
    hdr = f.read(30)
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    return info.header_offset + 30 + name_len + extra_len


def parallel_download(url, start, size, path, workers=8, piece=16 << 20):
    from concurrent.futures import ThreadPoolExecutor
    with open(path, "wb") as f:
        f.truncate(size)
    pieces = [(o, min(size, o + piece) - 1) for o in range(0, size, piece)]
    done = [0]

    def job(rng):
        a, b = rng
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={
                    "Range": "bytes=%d-%d" % (start + a, start + b), "User-Agent": UA})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) != b - a + 1:
                    raise IOError("short read")
                with open(path, "r+b") as f:
                    f.seek(a)
                    f.write(data)
                done[0] += 1
                print("    %s piece %d/%d" % (os.path.basename(path), done[0], len(pieces)),
                      flush=True)
                return
            except Exception as e:  # retry with back-off
                time.sleep(5 * (attempt + 1))
                last = e
        raise RuntimeError("download failed: %s" % last)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, pieces))


def inflate_file(src, dst):
    d = zlib.decompressobj(-15)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            buf = fi.read(8 << 20)
            if not buf:
                break
            fo.write(d.decompress(buf))
        fo.write(d.flush())


def extract_wanted(zi, seq, out_root):
    for want in WANTED:
        names = [n for n in zi.namelist() if n.endswith(want)]
        if not names:
            print("  %s: %s missing" % (seq, want))
            continue
        data = zi.read(names[0])
        dst = os.path.join(out_root, seq, want)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        print("  %s: %s (%.1f MB)" % (seq, want, len(data) / 1e6), flush=True)


def fetch(seq, out_root, workers=8):
    group = "machine_hall" if seq.startswith("MH") else (
        "vicon_room1" if seq.startswith("V1") else "vicon_room2")
    url = ARCHIVES[group]
    outer = HttpRangeFile(url)
    zo = zipfile.ZipFile(outer)
    inner_name = [n for n in zo.namelist() if n.endswith("%s/%s.zip" % (seq, seq))][0]
    info = zo.getinfo(inner_name)
    off = member_data_offset(outer, info)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        # The inner zip is deflated inside the outer one, so it cannot be
        # range-read. Download its compressed bytes with parallel range
        # requests, inflate locally, extract the CSVs, delete the rest.
        tmp_dir = os.path.join(out_root, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        comp_path = os.path.join(tmp_dir, seq + ".deflate")
        zip_path = os.path.join(tmp_dir, seq + ".zip")
        parallel_download(url, off, info.compress_size, comp_path, workers=workers)
        inflate_file(comp_path, zip_path)
        os.remove(comp_path)
        with zipfile.ZipFile(zip_path) as zi:
            extract_wanted(zi, seq, out_root)
        os.remove(zip_path)
        return
    inner = HttpRangeFile(url, base=off, size=info.file_size, block=1 << 20)
    zi = zipfile.ZipFile(inner)
    for want in WANTED:
        names = [n for n in zi.namelist() if n.endswith(want)]
        if not names:
            print("  %s: %s missing" % (seq, want))
            continue
        inf = zi.getinfo(names[0])
        # read the compressed bytes in one request, then decompress locally
        doff = member_data_offset(inner, inf)
        inner.seek(doff)
        comp = inner.read(inf.compress_size)
        if inf.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(comp, -15)
        elif inf.compress_type == zipfile.ZIP_STORED:
            data = comp
        else:
            raise RuntimeError("unsupported compression %d" % inf.compress_type)
        dst = os.path.join(out_root, seq, want)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(data)
        print("  %s: %s (%.1f MB)" % (seq, want, len(data) / 1e6))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seqs", nargs="+")
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "raw"))
    ap.add_argument("--workers", type=int, default=8, help="parallel range requests")
    args = ap.parse_args()
    for s in args.seqs:
        try:
            fetch(s, args.out, args.workers)
        except Exception as e:  # keep going with the other sequences
            print("  %s: FAILED (%s)" % (s, e), file=sys.stderr)


if __name__ == "__main__":
    main()
