# -*- coding: utf-8 -*-
"""Build the checkpoint census: exact decomposition of every scorer we dumped.

Reads `results/dump_*.json` (one file per dump batch, written by
`cafe/dsanet_dump.py` and `cafe/dsanet_xd_dump.py`) and writes
`results/table1_all.json`: theta, A, B, B|z, C and the two nuisance summaries
for every scorer in the census.

Two files in `results/` look similar and are not the same table:

* `table1_all.json` (this script) -- the full census, 20 rows: eleven UCF arms,
  the voided `ucf_motion9.72v_s234` arm, the extra flag variants, and four XD
  rows from an EARLIER XD pipeline.  The XD rows are kept only as provenance for
  the version census in Section 6.1; their theta (~0.53) is wrong because that
  pipeline aligned snippet-level scores against frame-level ground truth.  The
  paper's XD numbers come from `results/xd_table_raw.json` (theta 0.937-0.954).
* `table1.json` -- the twelve rows the paper actually uses: eleven UCF arms plus
  the voided arm, from the final pipeline.  The analyses that quote the paper
  (`cafe_rho_ci.py`, `cafe_perm.py`, `cafe_rho_ext.py`) read this file and
  select eleven rows from it.

Everything here is a pure function of the dumps, so the census can be rebuilt
without a GPU; only the dumps themselves need one.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

W = dict(A=0.631653, B=0.003880, C=0.364467)   # UCF-Crime test, data property


def load():
    rows = {}
    for f in glob.glob(os.path.join(RES, "dump_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if "d2" in v:
                rows[k] = v
    return rows


def main():
    rows = load()
    out = []
    for k, v in rows.items():
        d = v["d2"]
        out.append(dict(arm=k, theta=d["theta"], A=d["A"], B=d["B"],
                        Bz=d["B_cond"], C=d["C"],
                        within=v.get("within_mean2"),
                        corr=v.get("corr_motion2")))
    out.sort(key=lambda r: -r["theta"])
    print("%-24s %7s %7s %7s %7s %7s %7s %7s" % (
        "scorer", "theta", "A", "B", "B|z", "C", "within", "corr"))
    for r in out:
        print("%-24s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.3f" % (
            r["arm"], r["theta"], r["A"], r["B"], r["Bz"], r["C"],
            r["within"], r["corr"]))
    th = np.array([r["theta"] for r in out])
    bz = np.array([r["Bz"] for r in out])
    b = np.array([r["B"] for r in out])
    a = np.array([r["A"] for r in out])
    print("\nn=%d  theta range %.4f  B|z range %.4f" % (
        len(out), np.ptp(th), np.ptp(bz)))
    for nm, x in [("A", a), ("B", b), ("B|z", bz)]:
        rho, p = spearmanr(th, x)
        print("  spearman(theta, %-4s) = %+.3f (p=%.4f)" % (nm, rho, p))
    # how much of theta is the localization term, per model
    print("\nshare of theta carried by the same-video (localization) term:")
    for r in out:
        share = W["B"] * r["B"] / r["theta"]
        print("  %-24s theta %.4f -> B contributes %.5f (%.2f%%)"
              % (r["arm"], r["theta"], W["B"] * r["B"], 100 * share))
    path = os.path.join(RES, "table1_all.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
