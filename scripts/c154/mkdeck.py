#!/usr/bin/env python3
"""c154: derive a deck. mkdeck.py BASE OUT block/key=value [...]: replace key's value in <block> (the line keeps
its comment), or append the key to the block if absent ('block/key=' with no value deletes it). The source
deck is never modified. Note at the end which edits were made (header comment)."""
import re
import sys

base, out = sys.argv[1], sys.argv[2]
lines = open(base).read().split('\n')
notes = []
for arg in sys.argv[3:]:
    bk, val = arg.split('=', 1)
    blk, key = bk.split('/')
    start = next((n for n, l in enumerate(lines) if l.strip() == f'<{blk}>'), None)
    if start is None:
        lines += ['', f'<{blk}>']
        start = len(lines) - 1
    end = next((n for n in range(start + 1, len(lines)) if re.match(r'\s*<[A-Za-z0-9_]+>', lines[n])), len(lines))
    hit = [n for n in range(start + 1, end) if re.match(rf'\s*{re.escape(key)}\s*=', lines[n])]
    if val == '':
        for n in reversed(hit):
            del lines[n]
        notes.append(f'{blk}/{key} removed')
        continue
    if hit:
        n = hit[0]
        m = re.match(r'(\s*\S+\s*=\s*)(\S+)(.*)$', lines[n])
        lines[n] = f'{m.group(1)}{val}   # c154 (was {m.group(2)})'
    else:
        e = end
        while e > start + 1 and lines[e - 1].strip() == '':
            e -= 1
        lines.insert(e, f'{key} = {val}   # c154')
    notes.append(f'{blk}/{key}={val}')
hdr = f'# c154 deck: {base} with ' + '; '.join(notes)
open(out, 'w').write(hdr + '\n' + '\n'.join(lines))
print(out, '|', '; '.join(notes))
