#Requires -Version 5.1
<#
.SYNOPSIS
  gpu-run — machine-wide serialization of GPU-touching runs (one at a time).

.DESCRIPTION
  Acquires a cross-session named mutex (Global\gpu-run) as the mutual-exclusion
  primitive, records a metadata lockfile + contention log, runs the given command
  while HOLDING the lock (blocking on the child so the lock outlives the work),
  and releases everything on exit. The child's exit code becomes gpu-run's.

  WHY a named mutex, not just a lockfile
  (co-designed 2026-08-01 across fuel + baracuda + vulkane after the
  2026-07-31 host-aperture bugcheck; see docs/postmortems/):
    * A "stat then create" lockfile is TOCTOU — two wrappers both see no-file and
      both proceed into the GPU concurrently, the exact thing this prevents. A
      kernel mutex acquire is atomic.
    * The OS releases the mutex when the holder process dies — INCLUDING hard kill
      (taskkill / Stop-Process) and a machine bugcheck — so the next acquirer gets
      AbandonedMutexException and recovers deterministically and loudly, with no
      pid-liveness guessing and no stale-lockfile cleanup heuristic.
  The lockfile is metadata only (who holds it, since when, running what) plus a
  contention log ("who blocked whom, how long") — NOT the exclusion mechanism.

  SCOPE — wrap GPU-TOUCHING runs only: on-device tests, capture/replay,
  compute-sanitizer, cuBLAS/cuDNN, Vulkan device enumeration. Do NOT wrap
  compile-only (cargo check / clippy / build) — those never touch the device and
  serializing them only wastes wall-clock.

  THE HARNESS TAKES THE LOCK, NOT THE AGENT: call this from the test-runner
  wrapper so acquiring is structural, not something a human or agent must
  remember. A convention in a docs file is not a control (that is what failed).

  ASSUMPTIONS / LIMITS:
    * "Global\" is REQUIRED, not a bare name. A bare "gpu-run" resolves to
      "Local\gpu-run" — the per-logon-session namespace — so a process in another
      session (an elevated shell, a service, a scheduled task, a second
      RDP/console logon, or Docker Desktop's GPU helper) would get its OWN
      independent mutex under the same name and run concurrently, silently. The
      machine-wide "Global\" object is the whole point.
    * Participants are assumed to run as the same interactive user: %TEMP%
      resolves to one path for the metadata lockfile + log. (The "Global\" mutex
      is machine-wide regardless; only the lockfile/log path is per-user.)
    * NESTING: a nested gpu-run inside a child already under the lock would be a
      DIFFERENT process (different thread) and would DEADLOCK on the mutex its own
      parent holds. We export GPU_RUN_HELD into the child environment; a nested
      invocation detects it and PASSES THROUGH (logs, does not re-acquire).

.PARAMETER Project
  Label recorded in the lockfile: fuel | baracuda | vulkane | ...

.PARAMETER Command
  The command + args to run under the lock (everything after `--`).

.EXAMPLE
  pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-core --features cuda -- --ignored

.EXAMPLE
  pwsh scripts/gpu-run.ps1 -Project vulkane -- cargo test --workspace
#>
# No param() block AT ALL — deliberate. Under `pwsh -File`, an advanced binder
# chokes on the `--` separator ("parameter name '' is ambiguous"), and even a plain
# `param([string]$Project)` mis-binds the command's first token to $Project
# positionally ("parameter 'Project' specified more than once"). Parsing $args by
# hand is the only form that reliably survives `-File` invocation with an arbitrary
# child command — including the child's own `--flags` and a nested `--`. Convention:
# $Project is the FIRST argument. Usage: gpu-run.ps1 -Project <name> -- <cmd> <args...>

$ErrorActionPreference = 'Stop'

# The mutex name is a PROTOCOL CONSTANT shared by every participant across all
# projects and script versions. NEVER change it (not even to "-v2"): a different
# name means NO mutual exclusion between the two — both would run on the GPU at
# once, the exact failure this prevents (same trap as Local\ vs Global\). Version
# skew is surfaced by $GpuRunVersion (logged, not enforced), never by the name.
$MutexName     = 'Global\gpu-run'                   # MUST be Global\ (machine-wide), not Local\ — DO NOT VERSION
$GpuRunVersion = 2                                  # bump on behavioral change; written to the log + lockfile so drift is visible
$LockPath  = Join-Path $env:TEMP 'gpu-run.lock'     # metadata only, cleared on reboot (a lock must not survive one)
$LogPath   = Join-Path $env:TEMP 'gpu-run.log'

function Write-GpuLog([string] $msg) {
    $ts = (Get-Date).ToString('o')
    # Best-effort; never let a logging failure abort or wedge a GPU run.
    try { Add-Content -LiteralPath $LogPath -Value "$ts [$Project v$GpuRunVersion pid=$PID] $msg" -ErrorAction Stop } catch {}
}

# Parse the whole argument vector by hand (see the note at the top of the file).
# $Project must be the first argument; then an optional single `--` separator;
# the remainder is the command verbatim (its own `--flags` and any later `--`
# are preserved).
$argv    = @($args)
$Project = $null
if ($argv.Count -ge 2 -and $argv[0] -eq '-Project') {
    $Project = $argv[1]
    $argv    = @($argv | Select-Object -Skip 2)
}
if ($argv.Count -gt 0 -and $argv[0] -eq '--') {
    $argv = @($argv | Select-Object -Skip 1)
}
if ([string]::IsNullOrWhiteSpace($Project)) {
    throw 'gpu-run: -Project <name> must be the first argument. Usage: gpu-run.ps1 -Project <name> -- <cmd> <args...>'
}
if ($argv.Count -eq 0) {
    throw 'gpu-run: no command given. Usage: gpu-run.ps1 -Project <name> -- <cmd> <args...>'
}
$Command = $argv
$exe     = $Command[0]
$rest    = @($Command | Select-Object -Skip 1)
$cmdStr  = ($Command -join ' ')

# --- Nesting pass-through: already under the lock on this process tree? --------
if ($env:GPU_RUN_HELD -eq '1') {
    Write-GpuLog "nested gpu-run — passing through (lock already held by pid=$($env:GPU_RUN_HELD_PID)): $cmdStr"
    & $exe @rest
    exit $LASTEXITCODE
}

# --- Acquire the machine-wide mutex (the real mutual exclusion) ----------------
$mutex        = New-Object System.Threading.Mutex($false, $MutexName)
$acquired     = $false
$t0           = Get-Date
$holderBefore = if (Test-Path -LiteralPath $LockPath) {
                    (Get-Content -Raw -LiteralPath $LockPath -ErrorAction SilentlyContinue)
                } else { $null }

try {
    try {
        if ($mutex.WaitOne(0)) {
            $acquired = $true
        }
        else {
            Write-GpuLog "WAIT — GPU lock held by: $holderBefore"
            # A holder that DIES abandons the mutex (AbandonedMutexException, handled
            # below). But a holder that HANGS never abandons it — the exact
            # SIGKILL-worthy Vulkan-enumeration hang behind the 2026-07-31 incident —
            # so an UNBOUNDED WaitOne would silently wedge the whole queue, every
            # session looking idle while stuck (worse than no lock). Slice the wait,
            # warn to stderr each minute, and ABORT loud after the bound so a hang is
            # the noisiest state in the system, not the quietest.
            # GPU_RUN_TIMEOUT_MIN must EXCEED the longest legitimate GPU hold, or a
            # valid long job's waiters abort spuriously (exit 75) and it looks exactly
            # like the lock is broken. Default 45m fits a full Fuel CUDA suite; raise
            # it for longer jobs rather than let a real hold trip the guillotine.
            $timeoutMin = if ($env:GPU_RUN_TIMEOUT_MIN) { [int]$env:GPU_RUN_TIMEOUT_MIN } else { 45 }
            while (-not $acquired) {
                if ($mutex.WaitOne(60000)) { $acquired = $true; break }
                $mins      = [int]((Get-Date) - $t0).TotalMinutes
                $holderNow = if (Test-Path -LiteralPath $LockPath) {
                                 (Get-Content -Raw -LiteralPath $LockPath -ErrorAction SilentlyContinue)
                             } else { '(lockfile gone)' }
                $warn = "gpu-run: still WAITING ${mins}m for the GPU lock — holder: $holderNow"
                [Console]::Error.WriteLine($warn); Write-GpuLog $warn
                if ($mins -ge $timeoutMin) {
                    $abort = "gpu-run: ABORT after ${mins}m (>= ${timeoutMin}m) — holder appears WEDGED " +
                             "(a hung holder never abandons the mutex). Holder: $holderNow. " +
                             "Investigate/kill it (Stop-Process), then retry; raise the bound with " +
                             "`$env:GPU_RUN_TIMEOUT_MIN."
                    [Console]::Error.WriteLine($abort); Write-GpuLog $abort
                    exit 75   # EX_TEMPFAIL: distinct 'try again later', not a child code
                }
            }
        }
    }
    catch [System.Threading.AbandonedMutexException] {
        # Previous holder died mid-hold (hard kill / bugcheck). WaitOne STILL acquired.
        $acquired = $true
        Write-GpuLog "RECLAIMED abandoned GPU lock (previous holder died mid-hold). Stale lockfile: $holderBefore"
    }

    $waited = [int]((Get-Date) - $t0).TotalSeconds
    if ($holderBefore -and $waited -ge 1) { Write-GpuLog "acquired after ${waited}s wait" }

    # --- Metadata lockfile (NOT the exclusion primitive) ----------------------
    $meta = [ordered]@{
        v       = 1
        sv      = $GpuRunVersion
        pid     = $PID
        project = $Project
        cmd     = $cmdStr
        start   = (Get-Date).ToString('o')
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath $LockPath -Value $meta -Encoding UTF8
    Write-GpuLog "RUN: $cmdStr"

    # --- Run the child HOLDING the lock (blocks; lock outlives the work) -------
    $env:GPU_RUN_HELD     = '1'
    $env:GPU_RUN_HELD_PID = "$PID"
    # PS 7.4+ with EAP=Stop makes a FAILING native child throw instead of setting
    # $LASTEXITCODE; disable that locally so we capture the child's REAL exit code
    # (the finally still releases the lock either way).
    $PSNativeCommandUseErrorActionPreference = $false
    & $exe @rest
    $code = $LASTEXITCODE
    Write-GpuLog "DONE exit=$code : $cmdStr"
    exit $code
}
finally {
    $env:GPU_RUN_HELD     = $null
    $env:GPU_RUN_HELD_PID = $null
    if ($acquired) {
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
        try { $mutex.ReleaseMutex() } catch {}
        Write-GpuLog 'released'
    }
    $mutex.Dispose()
}
