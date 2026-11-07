# Plateau test launcher (COLDSTART.md, "Plateau test (11/5)"). Detached (Start-Process, hidden), Normal priority,
# 1 worker + 1 numba thread per run. Pass run names to (re)start only some, e.g.:
#   powershell -File launch_plateau.ps1 CatoA LivyB
# A run whose history.jsonl exists is RESUMED (same out dir, chunk numbering, DAgger files, RNG replay);
# otherwise it starts fresh from brains0/<name>_ck0.npz. Every run stops after chunk 200.
param([string[]]$Runs = @("CatoA", "CatoB", "CiceroA", "CiceroB", "LivyA", "LivyB", "CiceroLowA", "CiceroLowB"))
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $here "..\..\.venv-concepts\Scripts\python.exe"
$env:MARIO_STUCK = "40"  # plateau test was pre-registered at the old 2.7 s stuck limit (2026-11-06)
$env:NUMBA_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
foreach ($r in $Runs) {
    $out = "runs/$r"
    New-Item -ItemType Directory -Force (Join-Path $here $out) | Out-Null
    $a = @("ctrain.py", "--name", $r, "--init", "brains0/${r}_ck0.npz", "--out", $out, "--sel8", "--stop-chunk", "200",
           "--stop-at", '""', "--hours", "400", "--dagger-every", "6", "--episodes", "1", "--workers", "1", "--coach", "base")
    if (Test-Path (Join-Path $here "$out/history.jsonl")) { $a += "--resume" }
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $here -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $here "runs/$r.plateau.out") -RedirectStandardError (Join-Path $here "runs/$r.plateau.err")
    "$r pid $($p.Id)"
}
