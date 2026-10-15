"""Rerun everything, compare numbers against the old reports, scan for comment traces. Verdict in reports/VERIFY.txt."""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = r"D:\FAST\TrajLaw\sim"
OUT = []
NUM = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def toks(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        t = f.read()
    t = re.sub(r"[-+]?\d+\.\d+e[-+]?\d+", lambda m: "%.10g" % float(m.group()), t)
    return NUM.findall(t)


def main():
    run = "--run" in sys.argv
    scripts = ["e0", "e1", "e2", "e4", "e5", "e6"] + (["e3"] if "--all" in sys.argv else [])
    log("== 1 rerun each script (really executed: exit code and wall clock)")
    log("  %-6s %-8s %-10s %s" % ("script", "exit", "wall", "report"))
    for s in scripts:
        if run:
            t0 = time.time()
            r = subprocess.run([sys.executable, os.path.join(ROOT, "exp", s + ".py")],
                               cwd=ROOT, capture_output=True, text=True)
            el = time.time() - t0
            code = r.returncode
            tail = (r.stderr or "").strip().splitlines()[-1:] if code else []
        else:
            el, code, tail = float("nan"), "-", []
        rep = os.path.join(ROOT, "reports", s.upper() + ".txt")
        log("  %-6s %-8s %-10s %s %s" % (s, code, "%.1fs" % el if run else "(not run)",
                                         rep if os.path.exists(rep) else "missing", tail))

    log()
    log("== 2 numerical comparison against the old reports (numeric tokens from both sides, compared as sets)")
    log("  %-10s %-12s %-12s %-12s %s" % ("report", "old tokens", "new tokens", "shared", "only old / only new"))
    for s in ("E0", "E1", "E2", "E5", "E6"):
        po = os.path.join(OLD, "%s%s_report.txt" % (s, "_pairsim" if s == "E0" else ""))
        if s == "E1":
            po = os.path.join(OLD, "E1_report.txt")
        if s == "E2":
            po = os.path.join(OLD, "E2_report.txt")
        pn = os.path.join(ROOT, "reports", s + ".txt")
        a, b = toks(po), toks(pn)
        if a is None or b is None:
            log("  %-10s %s" % (s, "file missing"))
            continue
        sa, sb = set(a), set(b)
        only_a = sorted(sa - sb)[:6]
        only_b = sorted(sb - sa)[:6]
        log("  %-10s %-12d %-12d %-12d %s / %s"
            % (s, len(a), len(b), len(sa & sb), " ".join(only_a) or "-", " ".join(only_b) or "-"))

    log()
    log("== 3 comment and file-header scan")
    log("  %-14s %-10s %-10s %s" % ("file", "lines", "comment lines", "header block"))
    for d in ("tl", "exp", "tests", "bench"):
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if not f.endswith(".py"):
                continue
            p = os.path.join(ROOT, d, f)
            with open(p, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            nc = sum(1 for l in lines if l.strip().startswith("#"))
            doc = 0
            i = 0
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines) and lines[i].lstrip().startswith('"""'):
                doc = 1
            log("  %-14s %-10d %-10d %s" % ("%s/%s" % (d, f), len(lines), nc, "yes" if doc else "-"))

    log()
    log("== 4 peak GPU memory (torch side, this process)")
    log("  See the peak-memory line at the end of each experiment report; E0/E4/E3 all print it.")

    with open(os.path.join(ROOT, "reports", "VERIFY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
