#!/usr/bin/env python3
"""c187 (R-223 item 3): the production settings, in the deck (not code defaults):
<cfc> mg_threshold = mg_poisson_threshold = 1e-8 (rms, already), mg_min_vcycles = 1, mg_regrid_norm = max,
mg_regrid_threshold = 1e-7, mg_log = 1; <dyn_grmhd> tau_source_K = kcfc_bg. The DE block is unchanged.
    mkprod.py IN OUT 'header line'"""
import re, sys
src, out, hdr = sys.argv[1:4]
s = open(src).read()
cfc = ("# R-223 production settings (the user, 2026-11-03; C-187): converged CFC solves\n"
       "mg_min_vcycles = 1        # rms 1e-8 + one V-cycle per solve (C-160)\n"
       "mg_regrid_norm = max      # the post-regrid re-solve on the max norm (C-183)\n"
       "mg_regrid_threshold = 1.0e-7\n"
       "mg_log = 1                # MGLOG per solve\n")
for k in ('mg_min_vcycles', 'mg_regrid_norm', 'mg_regrid_threshold', 'mg_log', 'tau_source_K'):
    assert not re.search(rf'^\s*{k}\s*=', s, re.M), k
assert re.search(r'^mg_threshold\s*=\s*1\.0e-8', s, re.M) and re.search(r'^mg_poisson_threshold\s*=\s*1\.0e-8', s, re.M)
assert re.search(r'^mg_norm\s*=\s*rms', s, re.M)
s = re.sub(r'^<cfc>[^\n]*\n', lambda m: m.group(0) + cfc, s, count=1, flags=re.M)
dg = ("<dyn_grmhd>\n# R-223 production settings (C-187): Valencia's tau source with the trumpet background analytic (C-171)\n"
      "tau_source_K = kcfc_bg\n\n")
assert '<dyn_grmhd>' not in s
s = re.sub(r'^<cfc>', dg + '<cfc>', s, count=1, flags=re.M)
open(out, 'w').write('# ' + hdr + '\n' + s)
