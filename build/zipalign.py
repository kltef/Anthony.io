#!/usr/bin/env python3
"""
Align STORED ZIP entries to 4-byte boundaries by padding the extra field
in local file headers. Required for Android 11+ (targetSdkVersion >= 30).
Reads raw (possibly compressed) bytes directly so DEFLATED entries are
preserved as-is; only the local file header extra field is padded.
Usage: python3 zipalign.py in.apk out.apk
"""
import struct, zipfile, sys

def zipalign(in_path, out_path, align=4):
    with open(in_path, 'rb') as rf:
        raw = rf.read()

    with zipfile.ZipFile(in_path, 'r') as zin:
        entries = zin.infolist()

    buf = bytearray()
    new_offsets = {}

    for info in entries:
        fname = info.filename.encode('utf-8')
        pos   = len(buf)
        new_offsets[info.filename] = pos

        # Read raw (compressed) bytes directly from the source file
        lh_off    = info.header_offset
        fn_len    = struct.unpack_from('<H', raw, lh_off + 26)[0]
        ex_len    = struct.unpack_from('<H', raw, lh_off + 28)[0]
        data_off  = lh_off + 30 + fn_len + ex_len
        data      = raw[data_off : data_off + info.compress_size]

        # Compute padding needed so data starts on an `align`-byte boundary
        if info.compress_type == 0:  # STORED — must be aligned
            base = pos + 30 + len(fname)
            pad  = (align - (base % align)) % align
        else:
            pad = 0

        extra = b'\x00' * pad

        dt = info.date_time
        dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
        dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]

        hdr = struct.pack('<IHHHHHIIIHH',
            0x04034b50,
            info.create_version,
            info.flag_bits & ~0x8,
            info.compress_type,
            dostime, dosdate,
            info.CRC, info.compress_size, info.file_size,
            len(fname), len(extra))
        buf += hdr + fname + extra + data

    cd_start = len(buf)
    for info in entries:
        fname  = info.filename.encode('utf-8')
        offset = new_offsets[info.filename]
        dt = info.date_time
        dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
        dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
        cd = struct.pack('<IHHHHHHIIIHHHHHII',
            0x02014b50,
            (info.create_system << 8) | info.create_version,
            info.create_version,
            info.flag_bits & ~0x8,
            info.compress_type,
            dostime, dosdate,
            info.CRC, info.compress_size, info.file_size,
            len(fname), 0, 0, 0,
            info.internal_attr, info.external_attr, offset)
        buf += cd + fname

    cd_size = len(buf) - cd_start
    eocd = struct.pack('<IHHHHIIH',
        0x06054b50, 0, 0,
        len(entries), len(entries),
        cd_size, cd_start, 0)
    buf += eocd

    with open(out_path, 'wb') as f:
        f.write(buf)
    print(f"zipalign: {in_path} -> {out_path} ({len(buf)} bytes)")

if __name__ == '__main__':
    zipalign(sys.argv[1], sys.argv[2])
