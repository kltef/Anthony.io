#!/usr/bin/env python3
"""Extract State.io play-data straight from a Chromium WebView IndexedDB dump.

When the in-app export / adb backup / CDP paths are unavailable, we can still pull the
app's IndexedDB out of its sandbox (adb exec-out run-as ... tar) and decode it here.

The games live in the `statio_play` DB, `games` store, each stored value being the V8
structured-clone serialization of `{ts, bytes, game}`. The blocks are uncompressed, so we
scan the raw leveldb files for each record's top-object anchor (`o` + one-byte key "ts")
and run a minimal V8 deserializer from there — every value is self-terminating, so trailing
bytes don't matter and we never need a full LevelDB parser.

Usage:
  python tools/extract_idb_playdata.py <leveldb_dir> [-o out.json]
  # <leveldb_dir> holds 000NNN.log / 000NNN.ldb (…/IndexedDB/https_stateio.local_0.indexeddb.leveldb)
"""
import argparse
import glob
import json
import os
import struct

ANCHOR = b'o\x22\x02ts'   # kBeginJSObject, kOneByteString len=2, "ts"  -> the rec wrapper start


class V8Reader:
    """Decodes one V8-serialized value starting at a given offset. Only JSReceivers
    (objects/arrays) take reference ids, matching V8's id_map_ (strings/numbers are inline)."""
    def __init__(self, buf, pos):
        self.b = buf
        self.i = pos
        self.refs = []            # id -> object/array (JSReceivers only)

    def u8(self):
        v = self.b[self.i]; self.i += 1; return v

    def varint(self):
        shift = 0; result = 0
        while True:
            byte = self.b[self.i]; self.i += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                return result
            shift += 7

    def zigzag(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def read(self):
        tag = self.u8()
        c = chr(tag)
        if c == 'o':                                   # begin JS object
            obj = {}; self.refs.append(obj)
            while True:
                if self.b[self.i] == 0x7B:             # '{' end object
                    self.i += 1; self.varint()         # property count
                    return obj
                k = self.read(); v = self.read()
                obj[k] = v
        if c == 'A':                                   # begin dense array
            length = self.varint()
            arr = [None] * length; self.refs.append(arr)
            for j in range(length):
                arr[j] = self.read()
            # trailing named props (rare) then '$' + varint(props) + varint(length)
            while self.b[self.i] != 0x24:
                self.read(); self.read()
            self.i += 1; self.varint(); self.varint()
            return arr
        if c == 'a':                                   # begin sparse array
            self.varint()                              # length
            arr = {}; self.refs.append(arr)
            while self.b[self.i] != 0x40:              # '@' end sparse
                k = self.read(); v = self.read(); arr[k] = v
            self.i += 1; self.varint(); self.varint()
            # normalize to list if keys are 0..n-1
            try:
                n = max(int(x) for x in arr) + 1
                out = [None] * n
                for k, v in arr.items():
                    out[int(k)] = v
                return out
            except ValueError:
                return arr
        if c == 'I':                                   # int32 (zigzag)
            return self.zigzag()
        if c == 'U':                                   # uint32
            return self.varint()
        if c == 'N':                                   # double, 8 bytes LE
            v = struct.unpack_from('<d', self.b, self.i)[0]; self.i += 8
            return int(v) if v.is_integer() else v
        if c == '"':                                   # one-byte string
            n = self.varint(); s = self.b[self.i:self.i + n].decode('latin1'); self.i += n
            return s
        if c == 'c':                                   # two-byte string (utf16-le)
            n = self.varint(); s = self.b[self.i:self.i + n].decode('utf-16-le', 'replace'); self.i += n
            return s
        if c == 'S':                                   # utf8 string
            n = self.varint(); s = self.b[self.i:self.i + n].decode('utf-8', 'replace'); self.i += n
            return s
        if c == 'T':
            return True
        if c == 'F':
            return False
        if c == '0':                                   # null
            return None
        if c == '_':                                   # undefined
            return None
        if c == '-':                                   # the-hole
            return None
        if c == '^':                                   # object reference
            return self.refs[self.varint()]
        if c == 'D':                                   # Date (double ms)
            v = struct.unpack_from('<d', self.b, self.i)[0]; self.i += 8
            return v
        raise ValueError("unknown V8 tag 0x%02x (%r) at %d" % (tag, c, self.i - 1))


# ---- minimal LevelDB container readers (values only; blocks are uncompressed here) -------------
def uvarint(b, i):
    shift = 0; result = 0
    while True:
        byte = b[i]; i += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, i
        shift += 7


def snappy_decompress(data):
    """Minimal Snappy block decompression (LevelDB compresses SSTable blocks with it)."""
    _length, pos = uvarint(data, 0)
    out = bytearray()
    n = len(data)
    while pos < n:
        tag = data[pos]; pos += 1
        t = tag & 0x03
        if t == 0:                                   # literal run
            ln = tag >> 2
            if ln >= 60:
                extra = ln - 59
                ln = int.from_bytes(data[pos:pos + extra], 'little'); pos += extra
            ln += 1
            out += data[pos:pos + ln]; pos += ln
        else:                                        # back-reference copy
            if t == 1:
                ln = 4 + ((tag >> 2) & 0x07)
                offset = ((tag >> 5) << 8) | data[pos]; pos += 1
            elif t == 2:
                ln = 1 + (tag >> 2)
                offset = int.from_bytes(data[pos:pos + 2], 'little'); pos += 2
            else:
                ln = 1 + (tag >> 2)
                offset = int.from_bytes(data[pos:pos + 4], 'little'); pos += 4
            start = len(out) - offset
            for k in range(ln):                       # byte-by-byte (copies may overlap)
                out.append(out[start + k])
    return bytes(out)


def log_values(path):
    """LevelDB write-ahead log: strip 32KB block framing, reassemble WriteBatches, yield put values."""
    data = open(path, 'rb').read()
    BLOCK = 32768
    batches, frag, pos = [], b'', 0
    while pos + 7 <= len(data):
        if BLOCK - (pos % BLOCK) < 7:            # zero padding at block tail
            pos += BLOCK - (pos % BLOCK); continue
        length = int.from_bytes(data[pos + 4:pos + 6], 'little'); rtype = data[pos + 6]
        pos += 7
        payload = data[pos:pos + length]; pos += length
        if rtype == 1: batches.append(payload)               # FULL
        elif rtype == 2: frag = payload                      # FIRST
        elif rtype == 3: frag += payload                     # MIDDLE
        elif rtype == 4: batches.append(frag + payload); frag = b''  # LAST
    values = []
    for b in batches:                            # WriteBatch: seq(8) count(4) then entries
        if len(b) < 12: continue
        count = int.from_bytes(b[8:12], 'little'); i = 12
        for _ in range(count):
            if i >= len(b): break
            tag = b[i]; i += 1
            klen, i = uvarint(b, i); i += klen                # skip key
            if tag == 1:                                       # kTypeValue -> has a value
                vlen, i = uvarint(b, i); values.append(b[i:i + vlen]); i += vlen
            elif tag != 0:
                break
    return values


def ldb_values(path):
    """LevelDB SSTable (.ldb): footer -> index block -> data blocks -> yield entry values."""
    data = open(path, 'rb').read()
    footer = data[-48:]
    i = 0
    _mo, i = uvarint(footer, i); _ms, i = uvarint(footer, i)   # metaindex handle (ignored)
    io_, i = uvarint(footer, i); isz, i = uvarint(footer, i)   # index handle

    def block_contents(off, sz):
        raw = data[off:off + sz]; comp = data[off + sz]        # 5-byte trailer: 1 comp-type + 4 crc
        if comp == 1:
            raw = snappy_decompress(raw)
        return raw, (0 if comp == 1 else comp)

    def entries(block):
        num_restarts = int.from_bytes(block[-4:], 'little')
        end = len(block) - 4 - 4 * num_restarts
        out, i, key = [], 0, b''
        while i < end:
            shared, i = uvarint(block, i); nonshared, i = uvarint(block, i); vlen, i = uvarint(block, i)
            key = key[:shared] + block[i:i + nonshared]; i += nonshared
            out.append((key, block[i:i + vlen])); i += vlen
        return out

    idx, comp = block_contents(io_, isz)
    if comp != 0:
        return []
    values = []
    for _sep, handle in entries(idx):
        h = 0; off, h = uvarint(handle, h); sz, h = uvarint(handle, h)
        blk, bcomp = block_contents(off, sz)
        if bcomp != 0:                                         # snappy/zstd not expected here
            continue
        for _k, v in entries(blk):
            values.append(v)
    return values


def extract_file(path):
    vals = ldb_values(path) if path.endswith('.ldb') else log_values(path)
    games, errors = [], 0
    for v in vals:
        j = v.find(ANCHOR)                        # each value is now contiguous; anchor is the rec start
        if j < 0:
            continue
        try:
            rec = V8Reader(v, j).read()
            g = rec.get('game') if isinstance(rec, dict) else None
            if isinstance(g, dict) and 'moves' in g:
                games.append(g)
        except Exception:
            errors += 1
    return games, errors


def main():
    ap = argparse.ArgumentParser(description="Extract play-data from a WebView IndexedDB dump.")
    ap.add_argument("leveldb_dir")
    ap.add_argument("-o", "--out", default="phone_export/playdata_fader.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.leveldb_dir, "*.ldb")) +
                   glob.glob(os.path.join(args.leveldb_dir, "*.log")))
    all_games, total_err = [], 0
    for f in files:
        g, e = extract_file(f)
        total_err += e
        print("  %-16s %d records (%d decode errors)" % (os.path.basename(f), len(g), e))
        all_games += g

    # dedupe by timestamp (a game can't be written twice; guards any log/ldb overlap)
    seen, uniq = set(), []
    for g in all_games:
        key = (g.get('ts'), g.get('duration'), len(g.get('moves', [])))
        if key in seen:
            continue
        seen.add(key); uniq.append(g)
    uniq.sort(key=lambda g: g.get('ts', 0))

    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(uniq, f)
    print("\nExtracted %d games (%d raw, %d dup, %d errors) -> %s" % (
        len(uniq), len(all_games), len(all_games) - len(uniq), total_err, args.out))


if __name__ == "__main__":
    main()
