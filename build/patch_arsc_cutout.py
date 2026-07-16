#!/usr/bin/env python3
# Add android:windowLayoutInDisplayCutoutMode = shortEdges to the app theme inside resources.arsc.
#
# Why: the shell's theme (style 0x7f040000) inherits a framework *Fullscreen* parent (0x01030241).
# On phones with a camera cutout, a fullscreen window that does not opt into the cutout is
# LETTERBOXED below it — Android reserves the whole top band and paints it black (play-test
# 2026-07-13: top ~10% dead on two Samsungs + a Pixel, in every orientation; the in-page
# viewport-fit=cover fix could not help because the window never owned those pixels).
# LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES (=1) via attr windowLayoutInDisplayCutoutMode
# (0x01010586, android.R.attr = 16844166, API 28+; unknown attrs are ignored below API 28) lets
# the window extend into the cutout band; the page's env(safe-area-inset-top) CSS then keeps
# buttons out from under the camera hole. The two fixes compose.
#
#   usage: python3 build/patch_arsc_cutout.py <input.apk>   ->  build/resources.arsc (patched)
# build/repack.py picks up build/resources.arsc (if present) the same way it swaps the manifest.
import struct, sys, zipfile, os

ATTR_CUTOUT_MODE = 0x01010586   # android:windowLayoutInDisplayCutoutMode (16844166)
SHORT_EDGES = 1                  # LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
THEME_RESID = 0x7f040000         # the app theme referenced by the manifest's android:theme

def u16(b, o): return struct.unpack('<H', b[o:o+2])[0]
def u32(b, o): return struct.unpack('<I', b[o:o+4])[0]

def main():
    apk = sys.argv[1] if len(sys.argv) > 1 else 'State.io_v2.2.1-cutoutfix-testkey.apk'
    arsc = bytearray(zipfile.ZipFile(apk).read('resources.arsc'))

    # locate the package chunk and its type-string pool (to find the 'style' type id)
    off = 12
    pkg_off = None
    while off < len(arsc):
        t, sz = u16(arsc, off), u32(arsc, off + 4)
        if t == 0x0200:
            pkg_off = off
            break
        off += sz
    assert pkg_off is not None, 'package chunk not found'
    ts_off = u32(arsc, pkg_off + 8 + 4 + 256)
    pool_off = pkg_off + ts_off
    n = u32(arsc, pool_off + 8); str_start = u32(arsc, pool_off + 20); flags = u32(arsc, pool_off + 16)
    types = []
    for i in range(n):
        p = pool_off + str_start + u32(arsc, pool_off + 28 + 4 * i)
        if flags & 0x100:
            ln = arsc[p + 1]; types.append(arsc[p + 2:p + 2 + ln].decode('utf-8'))
        else:
            ln = u16(arsc, p); types.append(arsc[p + 2:p + 2 + ln * 2].decode('utf-16-le'))
    style_tid = types.index('style') + 1
    target_entry = THEME_RESID & 0xFFFF

    # find the TYPE chunk holding the style entry and patch it
    off = pkg_off + u16(arsc, pkg_off + 2)
    pkg_end = pkg_off + u32(arsc, pkg_off + 4)
    patched = False
    while off < pkg_end:
        t, hs, sz = u16(arsc, off), u16(arsc, off + 2), u32(arsc, off + 4)
        if t == 0x0201 and arsc[off + 8] == style_tid:
            entcount = u32(arsc, off + 12)
            entries_start = u32(arsc, off + 16)
            eoff = u32(arsc, off + hs + 4 * target_entry)
            assert eoff != 0xFFFFFFFF, 'style entry absent in this config'
            ep = off + entries_start + eoff
            esz, eflags = u16(arsc, ep), u16(arsc, ep + 2)
            assert eflags & 1, 'expected a complex (map) entry'
            cnt = u32(arsc, ep + 12)
            names = [u32(arsc, ep + 16 + 12 * i) for i in range(cnt)]
            if ATTR_CUTOUT_MODE in names:
                print('already patched'); patched = True; break
            # new map entry: name(4) size(2) res0(1) dataType(1) data(4); keep list sorted by attr id
            new_pair = struct.pack('<IHBBI', ATTR_CUTOUT_MODE, 8, 0, 0x10, SHORT_EDGES)
            insert_at = ep + 16 + 12 * cnt
            for i, nm in enumerate(names):
                if ATTR_CUTOUT_MODE < nm:
                    insert_at = ep + 16 + 12 * i
                    break
            arsc[insert_at:insert_at] = new_pair
            struct.pack_into('<I', arsc, ep + 12, cnt + 1)              # map count
            struct.pack_into('<I', arsc, off + 4, sz + 12)              # TYPE chunk size
            # entry offsets AFTER this entry shift by 12
            for e in range(entcount):
                eo = u32(arsc, off + hs + 4 * e)
                if eo != 0xFFFFFFFF and eo > eoff:
                    struct.pack_into('<I', arsc, off + hs + 4 * e, eo + 12)
            struct.pack_into('<I', arsc, pkg_off + 4, u32(arsc, pkg_off + 4) + 12)   # package size
            struct.pack_into('<I', arsc, 4, u32(arsc, 4) + 12)                        # table size
            patched = True
            print('patched: windowLayoutInDisplayCutoutMode=shortEdges added to style 0x%08x' % THEME_RESID)
            break
        off += sz
    assert patched, 'style TYPE chunk not found'
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources.arsc')
    open(out, 'wb').write(bytes(arsc))
    print('wrote', out, len(arsc), 'bytes')

if __name__ == '__main__':
    main()
