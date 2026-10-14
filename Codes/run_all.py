"""
run_all.py - reproduce every result, figure and table of the study from the public raw files.
Runs on Windows, macOS and Linux. About 8 hours with 4 workers on a laptop (13 hours on 2 cores); the independent
jobs of stage 3 run in parallel (--workers, each about 0.2-0.4 GB). All seeds are fixed, so the outputs equal the
archived results (check with compare_results.py).

Usage (from any folder):
  python Codes/run_all.py                  full run, 2 parallel workers
  python Codes/run_all.py --workers 4      more workers for the independent jobs
  python Codes/run_all.py --start 3        resume from a stage (0 data, 1 pre-specified, 2 revision, 3 main analyses,
                                           4 figures)
  python Codes/run_all.py --dry-run        list the steps without running them
Afterwards:
  python Codes/compare_results.py --ref <untouched copy of the archived results>

Progress is printed and written to logs/run_all_progress.log; each step writes its own log in logs/. If a step fails,
the run stops and prints the end of that step's log.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HEMIHOM_ROOT", CODE.parent))
LOGS = ROOT / "logs"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1",
           HEMIHOM_ROOT=str(ROOT))

# A step is (command, log file); the steps of a list run in parallel. "POOL" = the independent jobs below.
STAGES = {
    0: ("data, geometry and unit tests", [
        [("s1_data.py", "s1_data.log")],
        [("check.py tests", "check_tests.log")],
    ]),
    1: ("pre-specified analyses", [
        [("s2_prespecified.py B1", "s2_B1.log"), ("s2_prespecified.py B2", "s2_B2.log")],
        [("s2_prespecified.py B3", "s2_B3.log"), ("s2_prespecified.py B4", "s2_B4.log")],
        [("s2_prespecified.py real", "s2_real.log")],
        [("s2_prespecified.py rf_real 2 R:shuffle,R:spin_sym,SH:spin_sym", "s2_rf_real.log")],
        [("s2_prespecified.py rf_sim 2 300", "s2_rf_sim.log")],
        [("s2_prespecified.py summarize", "s2_summarize.log")],
        [("s2_prespecified.py baseline", "s2_baseline.log")],
        [("s2_prespecified.py partition", "s2_partition.log")],
    ]),
    2: ("first revision", [
        [("s2_prespecified.py rev_real", "s2_rev_real.log"), ("s2_prespecified.py rev_sim 77", "s2_rev_sim.log")],
        [("s2_prespecified.py rev_power", "s2_rev_power.log")],
        [("s2_prespecified.py percentiles", "s2_percentiles.log")],
        [("s2_prespecified.py rev_summarize", "s2_rev_summarize.log")],
        [("s2_prespecified.py checks", "s2_checks.log")],
    ]),
    3: ("main analyses", [
        [("s3_map_correlations.py sim", "s3_sim.log")],
        [("s3_map_correlations.py real", "s3_real.log")],
        [("s3_map_correlations.py approx", "s3_approx.log")],
        [("s4_cv_simulations.py dd_ols", "s4_dd_ols.log")],
        [("s4_cv_simulations.py estimand", "s4_estimand.log")],
        [("s4_cv_simulations.py cv lam77 0 500", "s4_cv_lam77.log"),
         ("s4_cv_simulations.py cv lam50 0 500", "s4_cv_lam50.log")],
        "POOL",
        [("s5_enigma.py rf_merge", "s5_rf_merge.log")],
        [("s6_statistics.py", "s6_statistics.log")],
    ]),
    4: ("figures and Source Data", [
        [("s7_figures.py", "s7_figures.log")],
    ]),
}
POOL = [  # longest first
    ("s4_cv_simulations.py glasser_cv 0 150", "s4_glasser_cv_0_150.log"),
    ("s4_cv_simulations.py glasser_cv 150 300", "s4_glasser_cv_150_300.log"),
    ("s5_enigma.py rf mainR 0 6", "s5_rf_mainR_0_6.log"), ("s5_enigma.py rf mainR 6 12", "s5_rf_mainR_6_12.log"),
    ("s5_enigma.py cv", "s5_cv.log"),
    ("s5_enigma.py rf mainSH 0 6", "s5_rf_mainSH_0_6.log"), ("s5_enigma.py rf mainSH 6 12", "s5_rf_mainSH_6_12.log"),
    ("s4_cv_simulations.py power 0", "s4_power_0.log"), ("s4_cv_simulations.py power 1", "s4_power_1.log"),
    ("s4_cv_simulations.py power 2", "s4_power_2.log"),
    ("s5_enigma.py rf alt 0 6", "s5_rf_alt_0_6.log"), ("s5_enigma.py rf alt 6 12", "s5_rf_alt_6_12.log"),
    ("s4_cv_simulations.py approx", "s4_approx.log"),
    ("s5_enigma.py noise", "s5_noise.log"), ("s5_enigma.py hyper", "s5_hyper.log"),
    ("s3_map_correlations.py gen_sim lam77_rho0.9 0 250", "s3_gen_sim09_a.log"),
    ("s3_map_correlations.py gen_sim lam77_rho0.9 250 500", "s3_gen_sim09_b.log"),
    ("s4_cv_simulations.py gen_cv 0 100", "s4_gen_cv_a.log"), ("s4_cv_simulations.py gen_cv 100 200", "s4_gen_cv_b.log"),
    ("s3_map_correlations.py gen_real", "s3_gen_real.log"),
    ("s3_map_correlations.py gen_sim lam77_rho0.0 0 250", "s3_gen_sim00.log"),
    ("s3_map_correlations.py gen_bilateral", "s3_gen_bilateral.log"),
    ("s4_cv_simulations.py half_cv avg", "s4_half_cv_avg.log"),
    ("s4_cv_simulations.py half_cv left", "s4_half_cv_left.log"),
    ("s3_map_correlations.py glasser", "s3_glasser.log"), ("s3_map_correlations.py half", "s3_half.log"),
    ("s5_enigma.py half", "s5_half.log"),
]


class StepFailed(Exception):
    pass


def say(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "run_all_progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail(path, n=15):
    try:
        return "\n    ".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def run_step(cmd, log, timing):
    args = cmd.split()
    say(f"  start {cmd}")
    t0 = time.time()
    with open(LOGS / log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, str(CODE / args[0])] + args[1:], cwd=CODE, env=ENV, stdout=f,
                           stderr=subprocess.STDOUT)
    dt = time.time() - t0
    timing.append((cmd, log, r.returncode, round(dt, 1)))
    if r.returncode != 0:
        raise StepFailed(f"{cmd} failed (exit code {r.returncode}); end of logs/{log}:\n    {tail(LOGS / log)}")
    say(f"  done  {cmd}  ({dt / 60:.1f} min)")


def run_parallel(steps, workers, timing):
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(steps)))) as ex:
        for fu in [ex.submit(run_step, cmd, log, timing) for cmd, log in steps]:
            try:
                fu.result()
            except StepFailed as e:
                errors.append(str(e))
                say("  FAILED " + str(e).splitlines()[0])
    if errors:
        raise StepFailed("\n".join(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HEMIHOM_WORKERS", "2")))
    ap.add_argument("--start", type=int, default=0, choices=sorted(STAGES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for k, (name, items) in STAGES.items():
            if k >= args.start:
                print(f"stage {k}: {name}")
                for it in items:
                    print(f"  {len(POOL)} independent jobs on {args.workers} workers: " + "; ".join(c for c, _ in POOL)
                          if it == "POOL" else "  " + "  ||  ".join(c for c, _ in it))
        return 0
    for d in ("logs", "results", "figures"):
        (ROOT / d).mkdir(exist_ok=True)
    if args.start == 0:   # files older than this time stamp were not regenerated by the run (compare_results.py)
        (LOGS / "run_all_started.txt").write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n",
                                                  encoding="utf-8")
    say(f"run_all.py: Python {sys.version.split()[0]} ({sys.executable}); {args.workers} workers; "
        f"from stage {args.start}; project folder {ROOT}")
    timing, t_all = [], time.time()
    try:
        for k, (name, items) in STAGES.items():
            if k < args.start:
                continue
            say(f"stage {k}: {name}")
            for it in items:
                if it == "POOL":
                    say(f"  {len(POOL)} independent jobs on {args.workers} workers")
                    run_parallel(POOL, args.workers, timing)
                elif len(it) == 1:
                    run_step(it[0][0], it[0][1], timing)
                else:
                    run_parallel(it, args.workers, timing)
    except StepFailed as e:
        say("STOPPED: " + str(e))
        say("Fix the cause and resume with: python Codes/run_all.py --start <stage of the failed step>")
        return 1
    finally:
        tf = LOGS / "run_all_timing.csv"   # appended, so resumed runs keep the earlier steps
        new = not tf.exists() or args.start == 0
        with open(tf, "w" if new else "a", encoding="utf-8") as f:
            if new:
                f.write("step,log,exit_code,seconds\n")
            for cmd, log, rc, dt in timing:
                f.write(f"\"{cmd}\",{log},{rc},{dt}\n")
    say(f"ALL DONE in {(time.time() - t_all) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
