# Colab helper: fetch IMU + ground truth of EuRoC sequences with parallel HTTP
# range requests (same idea as tools/fetch_euroc.py, tuned for a fast link).
#
#   python fetch_all.py MH_01_easy MH_02_easy ...      -> /content/pf/data/raw
#
# ETH rate-limits aggressively (HTTP 429 after ~2 sequences at 24 connections).
# Keep 8 workers, wait 60 s on 429 and 30 s between sequences.
import io, os, struct, sys, time, zipfile, zlib, urllib.request
from concurrent.futures import ThreadPoolExecutor

ARCH = {"MH": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe", "V1": "02ecda9a-298f-498b-970c-b7c44334d880",
        "V2": "ea12bc01-3677-4b4c-853d-87c7870b8c44"}
BASE = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/%s/content"
UA = "pathfinder-dataset-fetch/1.0"
WANT = ("mav0/imu0/data.csv", "mav0/state_groundtruth_estimate0/data.csv")
OUT = os.environ.get("PF_RAW", "/content/pf/data/raw")
WORKERS = 8


def rng(url, a, b, tries=12):
    err = None
    for k in range(tries):
        try:
            r = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b), "User-Agent": UA})
            d = urllib.request.urlopen(r, timeout=300).read()
            if len(d) == b - a + 1:
                return d
        except Exception as e:
            err = e
        time.sleep(60 if "429" in str(err) else 3 * (k + 1))
    raise RuntimeError("range failed %s" % err)


class RF(io.RawIOBase):
    def __init__(s, url):
        s.url = url; s.pos = 0; s.cache = {}
        r = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": UA})
        s.size = int(urllib.request.urlopen(r).headers["Content-Range"].split("/")[1])

    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.pos

    def seek(s, o, w=0):
        s.pos = o if w == 0 else (s.pos + o if w == 1 else s.size + o)
        return s.pos

    def read(s, n=-1):
        n = s.size - s.pos if n is None or n < 0 else min(n, s.size - s.pos)
        out = bytearray(); B = 1 << 20
        while n > 0:
            b = s.pos // B
            if b not in s.cache:
                s.cache[b] = rng(s.url, b * B, min(s.size, (b + 1) * B) - 1)
            o = s.pos - b * B; c = s.cache[b][o:o + n]; out += c; s.pos += len(c); n -= len(c)
        return bytes(out)

    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)


def fetch(seq):
    url = BASE % ARCH[seq[:2]]
    f = RF(url); zo = zipfile.ZipFile(f)
    info = zo.getinfo([n for n in zo.namelist() if n.endswith("%s/%s.zip" % (seq, seq))][0])
    f.seek(info.header_offset); h = f.read(30)
    off = info.header_offset + 30 + sum(struct.unpack("<HH", h[26:30]))
    P = 16 << 20
    pieces = [(o, min(info.compress_size, o + P) - 1) for o in range(0, info.compress_size, P)]
    tmp = os.path.join(os.path.dirname(OUT), "tmp_%s.zip" % seq)
    dec = zlib.decompressobj(-15)
    with open(tmp, "wb") as fo, ThreadPoolExecutor(WORKERS) as ex:
        for d in ex.map(lambda p: rng(url, off + p[0], off + p[1]), pieces):  # ordered
            fo.write(dec.decompress(d))
        fo.write(dec.flush())
    with zipfile.ZipFile(tmp) as zi:
        for w in WANT:
            n = [x for x in zi.namelist() if x.endswith(w)][0]
            dst = os.path.join(OUT, seq, w); os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(zi.read(n))
    os.remove(tmp)
    print(seq, "ok", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        t = time.time()
        time.sleep(30)
        try:
            fetch(s); print("  %.0fs" % (time.time() - t), flush=True)
        except Exception as e:
            print(s, "FAILED", e, flush=True)
