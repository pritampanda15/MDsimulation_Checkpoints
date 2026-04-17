#!/usr/bin/env python3
"""
MD Simulation Checkpoint Progress Tracker
==========================================
Tracks and reports progress of GROMACS MD simulations.
Supports both protein-in-water and protein-ligand setups.
Includes automatic resume of stopped/crashed mdrun processes.

Usage:
    python md_checkpoint_tracker.py --dir /path/to/sim --type protein_water
    python md_checkpoint_tracker.py --dir /path/to/sim --type protein_ligand --ligand LIG
    python md_checkpoint_tracker.py --dir /path/to/sim --watch --interval 60
    python md_checkpoint_tracker.py --dir /path/to/sim --auto-resume
    python md_checkpoint_tracker.py --dir /path/to/sim --auto-resume --ntmpi 1 --ntomp 16 --gpu-id 0

Author : Pritam Kumar Panda
Affil  : Stanford University
"""

import os
import re
import sys
import time
import glob
import signal
import shutil
import argparse
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationPhase:
    name: str
    mdp_key: str                 # prefix used in file names
    required_files: list[str]
    optional_files: list[str] = field(default_factory=list)
    completed: bool = False
    progress_pct: float = 0.0
    current_time_ns: float = 0.0
    total_time_ns: float = 0.0
    last_checkpoint: Optional[str] = None
    wall_time_elapsed: Optional[str] = None
    estimated_remaining: Optional[str] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EnergyStats:
    temperature: Optional[float] = None      # K
    pressure: Optional[float] = None         # bar
    potential_energy: Optional[float] = None # kJ/mol
    total_energy: Optional[float] = None     # kJ/mol
    density: Optional[float] = None          # kg/m³
    rmsd: Optional[float] = None             # nm (if available via gmx rms)


@dataclass
class SimulationReport:
    sim_type: str
    sim_dir: str
    generated_at: str
    phases: list[SimulationPhase]
    overall_progress_pct: float
    energy_stats: EnergyStats
    warnings: list[str]
    resume_commands: list[str]


@dataclass
class ResumeResult:
    phase_name: str
    phase_prefix: str
    checkpoint_used: str
    command: str
    success: bool
    pid: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ─────────────────────────────────────────────────────────────
# PHASE DEFINITIONS
# ─────────────────────────────────────────────────────────────

PROTEIN_WATER_PHASES = [
    SimulationPhase("Energy Minimisation",  "em",   ["em.tpr"],          ["em.log", "em.edr", "em.trr", "em.gro"]),
    SimulationPhase("NVT Equilibration",    "nvt",  ["nvt.tpr"],         ["nvt.log", "nvt.edr", "nvt.xtc", "nvt.cpt", "nvt.gro"]),
    SimulationPhase("NPT Equilibration",    "npt",  ["npt.tpr"],         ["npt.log", "npt.edr", "npt.xtc", "npt.cpt", "npt.gro"]),
    SimulationPhase("Production MD",        "md",   ["md.tpr"],          ["md.log", "md.edr", "md.xtc",  "md.cpt",  "md.gro"]),
]

PROTEIN_LIGAND_PHASES = [
    SimulationPhase("Energy Minimisation",  "em",   ["em.tpr"],          ["em.log", "em.edr", "em.trr", "em.gro"]),
    SimulationPhase("NVT Equilibration",    "nvt",  ["nvt.tpr"],         ["nvt.log", "nvt.edr", "nvt.xtc", "nvt.cpt", "nvt.gro"]),
    SimulationPhase("NPT Equilibration",    "npt",  ["npt.tpr"],         ["npt.log", "npt.edr", "npt.xtc", "npt.cpt", "npt.gro"]),
    SimulationPhase("Production MD",        "md",   ["md.tpr"],          ["md.log", "md.edr", "md.xtc",  "md.cpt",  "md.gro"]),
]

# ─────────────────────────────────────────────────────────────
# PARSER UTILITIES
# ─────────────────────────────────────────────────────────────

def parse_log_progress(log_path: str) -> tuple[float, float, str | None, str | None]:
    """
    Parse GROMACS .log file to extract simulation time, total time,
    wall-time elapsed and estimated time remaining.

    Returns: (current_time_ps, total_time_ps, elapsed_str, remaining_str)
    """
    current_ps = 0.0
    total_ps = 0.0
    elapsed_str = None
    remaining_str = None

    if not os.path.isfile(log_path):
        return current_ps, total_ps, elapsed_str, remaining_str

    try:
        with open(log_path, "r", errors="replace") as fh:
            content = fh.read()

        # Total simulation time from nsteps × dt
        nsteps_m = re.search(r"nsteps\s*=\s*(\d+)", content)
        dt_m      = re.search(r"dt\s*=\s*([\d.eE+\-]+)", content)
        if nsteps_m and dt_m:
            total_ps = int(nsteps_m.group(1)) * float(dt_m.group(1))

        # Latest "Step / Time" line  →  current time in ps
        step_times = re.findall(r"^\s*Step\s+Time\s*\n\s*(\d+)\s+([\d.]+)", content, re.MULTILINE)
        if step_times:
            current_ps = float(step_times[-1][1])

        # Wall-clock from "Performance:" block
        perf_m = re.search(r"Wall time \(real\)\s*=\s*([\d.]+)\s*s", content)
        if perf_m:
            elapsed_s = float(perf_m.group(1))
            elapsed_str = str(timedelta(seconds=int(elapsed_s)))
            if total_ps > 0 and current_ps > 0:
                frac = current_ps / total_ps
                remaining_s = elapsed_s * (1 - frac) / frac if frac > 0 else 0
                remaining_str = str(timedelta(seconds=int(remaining_s)))

        # Fallback: last "Elapsed wall time" line
        if elapsed_str is None:
            ew = re.findall(r"Elapsed wall time\s*:\s*([\d.]+)\s*s", content)
            if ew:
                elapsed_str = str(timedelta(seconds=int(float(ew[-1]))))

    except Exception as exc:
        pass  # non-fatal

    return current_ps, total_ps, elapsed_str, remaining_str


def parse_energy_from_log(log_path: str) -> EnergyStats:
    """Extract last reported energy/temperature/pressure from .log."""
    stats = EnergyStats()
    if not os.path.isfile(log_path):
        return stats

    try:
        with open(log_path, "r", errors="replace") as fh:
            content = fh.read()

        def last_value(pattern):
            hits = re.findall(pattern, content, re.MULTILINE)
            return float(hits[-1]) if hits else None

        stats.temperature      = last_value(r"Temperature\s+([\d.\-eE+]+)")
        stats.pressure         = last_value(r"Pressure \(bar\)\s+([\d.\-eE+]+)")
        stats.potential_energy = last_value(r"Potential Energy\s+([\d.\-eE+]+)")
        stats.total_energy     = last_value(r"Total Energy\s+([\d.\-eE+]+)")
        stats.density          = last_value(r"Density\s+([\d.\-eE+]+)")
    except Exception:
        pass

    return stats


def validate_checkpoint(gmx: str, cpt_path: str) -> bool:
    """
    Run 'gmx check -f <cpt>' to verify the checkpoint is readable.
    Returns True if valid, False if corrupted.
    """
    try:
        result = subprocess.run(
            [gmx, "check", "-f", cpt_path],
            capture_output=True, text=True, timeout=30,
        )
        # gmx check exits 0 and prints "Reading checkpoint file" on success
        return result.returncode == 0 and "Reading checkpoint" in (result.stdout + result.stderr)
    except Exception:
        return False


def find_best_checkpoint(gmx: str, sim_dir: str, prefix: str) -> Optional[str]:
    """
    Return the best usable checkpoint:
    1. Try <prefix>.cpt  (most recent)
    2. Fall back to <prefix>_prev.cpt  if primary is corrupted
    3. Return None if neither is valid
    """
    primary  = os.path.join(sim_dir, f"{prefix}.cpt")
    fallback = os.path.join(sim_dir, f"{prefix}_prev.cpt")

    for cpt in (primary, fallback):
        if os.path.isfile(cpt):
            if validate_checkpoint(gmx, cpt):
                return cpt
            else:
                print(f"  {YELLOW}⚠  {os.path.basename(cpt)} appears corrupted, trying fallback…{RESET}")
    return None


def is_mdrun_running(sim_dir: str, prefix: str) -> bool:
    """
    Check whether an mdrun process is already writing to this directory.
    Heuristics:
      1. GROMACS lock file  (#<prefix>.lock or .#<prefix>.log)
      2. Process list scan for 'mdrun' with matching -deffnm
    """
    # Lock file check
    for lock_name in (f"#{prefix}.lock", f".#{prefix}.log"):
        if os.path.isfile(os.path.join(sim_dir, lock_name)):
            return True

    # Process scan (Linux only)
    try:
        result = subprocess.run(
            ["pgrep", "-a", "-f", f"mdrun.*{prefix}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except FileNotFoundError:
        pass  # pgrep not available on all systems

    return False


def setup_resume_logger(sim_dir: str) -> logging.Logger:
    """Configure a file logger that appends to resume.log in sim_dir."""
    log_path = os.path.join(sim_dir, "md_resume.log")
    logger = logging.getLogger("md_resume")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
        logger.addHandler(fh)
    return logger



    """Return path of the most recent .cpt file for a given prefix."""
    candidates = sorted(
        glob.glob(os.path.join(sim_dir, f"{prefix}*.cpt")),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def file_age_str(fpath: str) -> str:
    """Human-readable age of a file."""
    if not os.path.isfile(fpath):
        return "N/A"
    age_s = time.time() - os.path.getmtime(fpath)
    return str(timedelta(seconds=int(age_s))) + " ago"


def gmx_binary() -> str:
    """Resolve the GROMACS binary (gmx or gmx_mpi)."""
    for candidate in ("gmx", "gmx_mpi", "gmx_d"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return "gmx"   # fallback – may not be on PATH


# ─────────────────────────────────────────────────────────────
# CORE CHECKER
# ─────────────────────────────────────────────────────────────

class MDCheckpointTracker:
    """
    Main tracker class.  Inspect a GROMACS simulation directory and
    assemble a full SimulationReport.
    """

    def __init__(
        self,
        sim_dir: str,
        sim_type: str = "protein_water",
        ligand_name: str = "LIG",
        gmx: Optional[str] = None,
        ntmpi: int = 1,
        ntomp: int = 8,
        gpu_id: str = "0",
        pin: bool = True,
    ):
        self.sim_dir     = os.path.abspath(sim_dir)
        self.sim_type    = sim_type
        self.ligand_name = ligand_name
        self.gmx         = gmx or gmx_binary()
        self.ntmpi       = ntmpi
        self.ntomp       = ntomp
        self.gpu_id      = gpu_id
        self.pin         = pin
        self.logger      = setup_resume_logger(self.sim_dir)

        template_phases = (
            PROTEIN_LIGAND_PHASES if sim_type == "protein_ligand"
            else PROTEIN_WATER_PHASES
        )
        import copy
        self.phases: list[SimulationPhase] = copy.deepcopy(template_phases)

    # ── public ──────────────────────────────────────────────

    def run(self) -> SimulationReport:
        warnings: list[str] = []
        energy_stats = EnergyStats()

        for phase in self.phases:
            self._check_phase(phase, warnings)
            if phase.progress_pct > 0:
                # Use the last active phase for energy stats
                log_path = os.path.join(self.sim_dir, f"{phase.mdp_key}.log")
                energy_stats = parse_energy_from_log(log_path)

        overall = self._overall_progress()
        resume_cmds = self._build_resume_commands()

        return SimulationReport(
            sim_type=self.sim_type,
            sim_dir=self.sim_dir,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            phases=self.phases,
            overall_progress_pct=overall,
            energy_stats=energy_stats,
            warnings=warnings,
            resume_commands=resume_cmds,
        )

    # ── private ─────────────────────────────────────────────

    def _check_phase(self, phase: SimulationPhase, warnings: list[str]):
        prefix    = phase.mdp_key
        sim_dir   = self.sim_dir
        log_path  = os.path.join(sim_dir, f"{prefix}.log")
        tpr_path  = os.path.join(sim_dir, f"{prefix}.tpr")
        gro_path  = os.path.join(sim_dir, f"{prefix}.gro")
        cpt_path  = find_best_checkpoint(self.gmx, sim_dir, prefix)

        # ── not started ────────────────────────────────────
        if not os.path.isfile(tpr_path):
            phase.notes.append("TPR not found – phase not prepared yet.")
            return

        # ── check completion ───────────────────────────────
        if os.path.isfile(gro_path) and os.path.isfile(log_path):
            with open(log_path, "r", errors="replace") as fh:
                tail = fh.read()[-4096:]
            if "Finished mdrun" in tail or "GROMACS reminds you" in tail:
                phase.completed     = True
                phase.progress_pct  = 100.0
                phase.notes.append("Phase completed successfully.")

        # ── parse progress from log ────────────────────────
        current_ps, total_ps, elapsed, remaining = parse_log_progress(log_path)
        if total_ps > 0:
            phase.current_time_ns   = round(current_ps / 1000, 4)
            phase.total_time_ns     = round(total_ps  / 1000, 4)
            phase.progress_pct      = round(100 * current_ps / total_ps, 2)
            phase.wall_time_elapsed = elapsed
            phase.estimated_remaining = remaining

        # ── checkpoint info ────────────────────────────────
        if cpt_path:
            phase.last_checkpoint = f"{os.path.basename(cpt_path)}  ({file_age_str(cpt_path)})"

        # ── missing optional files ─────────────────────────
        for fname in phase.optional_files:
            fpath = os.path.join(sim_dir, fname)
            if not os.path.isfile(fpath):
                phase.notes.append(f"Optional file missing: {fname}")

        # ── stale checkpoint warning ───────────────────────
        if cpt_path and not phase.completed:
            age_s = time.time() - os.path.getmtime(cpt_path)
            if age_s > 3600:
                warnings.append(
                    f"[{phase.name}] Checkpoint is {timedelta(seconds=int(age_s))} old – "
                    "simulation may be stalled."
                )

        # ── energy/pressure sanity ─────────────────────────
        stats = parse_energy_from_log(log_path)
        if stats.temperature is not None:
            if stats.temperature < 200 or stats.temperature > 400:
                warnings.append(
                    f"[{phase.name}] Temperature {stats.temperature:.1f} K looks unusual."
                )
        if stats.pressure is not None:
            if abs(stats.pressure) > 500:
                warnings.append(
                    f"[{phase.name}] Pressure {stats.pressure:.1f} bar is very high – check NPT settings."
                )

    def _overall_progress(self) -> float:
        """Weight each phase equally."""
        n = len(self.phases)
        if n == 0:
            return 0.0
        return round(sum(p.progress_pct for p in self.phases) / n, 2)

    def _build_resume_cmd_parts(self, phase: SimulationPhase, cpt_path: str) -> list[str]:
        """Return the mdrun command as a list of tokens."""
        cmd = [
            self.gmx, "mdrun", "-v",
            "-deffnm",  phase.mdp_key,
            "-cpi",     os.path.basename(cpt_path),
            "-ntmpi",   str(self.ntmpi),
            "-ntomp",   str(self.ntomp),
        ]
        if self.gpu_id:
            cmd += ["-gpu_id", self.gpu_id]
        if self.pin:
            cmd += ["-pin", "on"]
        return cmd

    def _build_resume_commands(self) -> list[str]:
        cmds = []
        for phase in self.phases:
            if phase.completed or phase.progress_pct == 0.0:
                continue
            cpt = find_best_checkpoint(self.gmx, self.sim_dir, phase.mdp_key)
            if cpt:
                parts = self._build_resume_cmd_parts(phase, cpt)
                cmds.append(f"# Resume {phase.name}\n" + " ".join(parts))
        return cmds

    # ── AUTO RESUME ─────────────────────────────────────────

    def auto_resume(self, dry_run: bool = False) -> list[ResumeResult]:
        """
        Find all incomplete phases, validate their checkpoints,
        and resume each one sequentially (phases are serial by nature).

        Args:
            dry_run: If True, print the command but do not execute.

        Returns:
            List of ResumeResult for each phase that was attempted.
        """
        results: list[ResumeResult] = []

        incomplete = [
            p for p in self.phases
            if not p.completed and p.progress_pct > 0.0
        ]

        if not incomplete:
            # Nothing mid-run — check if any phase hasn't started but
            # a previous phase just finished (pick up the next one)
            completed_keys = {p.mdp_key for p in self.phases if p.completed}
            for phase in self.phases:
                if not phase.completed and phase.progress_pct == 0.0:
                    tpr = os.path.join(self.sim_dir, f"{phase.mdp_key}.tpr")
                    if os.path.isfile(tpr):
                        # Phase is prepared but never started — start fresh
                        incomplete = [phase]
                        break

        if not incomplete:
            print(f"\n{GREEN}  No incomplete phases found. Nothing to resume.{RESET}\n")
            return results

        for phase in incomplete:
            print(f"\n{CYAN}{'─'*60}{RESET}")
            print(f"  {BOLD}Attempting resume: {phase.name}{RESET}")

            # ── already running? ──────────────────────────
            if is_mdrun_running(self.sim_dir, phase.mdp_key):
                msg = f"mdrun already running for '{phase.mdp_key}' — skipping."
                print(f"  {YELLOW}⚠  {msg}{RESET}")
                self.logger.warning(msg)
                results.append(ResumeResult(
                    phase_name=phase.name, phase_prefix=phase.mdp_key,
                    checkpoint_used="N/A", command="N/A",
                    success=False, error=msg,
                ))
                continue

            # ── find & validate checkpoint ────────────────
            if phase.progress_pct > 0.0:
                # Mid-run: must use checkpoint
                cpt = find_best_checkpoint(self.gmx, self.sim_dir, phase.mdp_key)
                if not cpt:
                    msg = f"No valid checkpoint found for '{phase.mdp_key}'. Cannot resume."
                    print(f"  {RED}✗  {msg}{RESET}")
                    self.logger.error(msg)
                    results.append(ResumeResult(
                        phase_name=phase.name, phase_prefix=phase.mdp_key,
                        checkpoint_used="NONE", command="N/A",
                        success=False, error=msg,
                    ))
                    continue
                cmd_parts = self._build_resume_cmd_parts(phase, cpt)
                cpt_used = os.path.basename(cpt)
            else:
                # Never started — fresh start (no -cpi)
                tpr = os.path.join(self.sim_dir, f"{phase.mdp_key}.tpr")
                cmd_parts = [
                    self.gmx, "mdrun", "-v",
                    "-deffnm", phase.mdp_key,
                    "-ntmpi",  str(self.ntmpi),
                    "-ntomp",  str(self.ntomp),
                ]
                if self.gpu_id:
                    cmd_parts += ["-gpu_id", self.gpu_id]
                if self.pin:
                    cmd_parts += ["-pin", "on"]
                cpt_used = "NONE (fresh start)"

            cmd_str = " ".join(cmd_parts)
            print(f"  {BLUE}CMD : {cmd_str}{RESET}")
            self.logger.info(f"Resuming {phase.name} | cpt={cpt_used} | cmd={cmd_str}")

            if dry_run:
                print(f"  {DIM}[dry-run] Command not executed.{RESET}")
                results.append(ResumeResult(
                    phase_name=phase.name, phase_prefix=phase.mdp_key,
                    checkpoint_used=cpt_used, command=cmd_str,
                    success=True, pid=None,
                ))
                continue

            # ── backup existing log ───────────────────────
            log_path = os.path.join(self.sim_dir, f"{phase.mdp_key}.log")
            if os.path.isfile(log_path):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = log_path.replace(".log", f"_backup_{ts}.log")
                shutil.copy2(log_path, backup)
                print(f"  {DIM}Log backed up → {os.path.basename(backup)}{RESET}")

            # ── launch mdrun ──────────────────────────────
            try:
                proc = subprocess.Popen(
                    cmd_parts,
                    cwd=self.sim_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,   # detach from current terminal
                )
                pid = proc.pid
                print(f"  {GREEN}✔  mdrun launched (PID {pid}){RESET}")
                self.logger.info(f"mdrun started PID={pid} for {phase.name}")

                results.append(ResumeResult(
                    phase_name=phase.name, phase_prefix=phase.mdp_key,
                    checkpoint_used=cpt_used, command=cmd_str,
                    success=True, pid=pid,
                ))

                # Wait a moment and confirm it's still alive
                time.sleep(3)
                if proc.poll() is not None:
                    rc = proc.returncode
                    err_msg = f"mdrun exited immediately with code {rc}. Check {phase.mdp_key}.log."
                    print(f"  {RED}✗  {err_msg}{RESET}")
                    self.logger.error(err_msg)
                    results[-1].success = False
                    results[-1].error   = err_msg
                else:
                    print(f"  {GREEN}   Process still alive after 3s ✔{RESET}")

            except Exception as exc:
                err_msg = str(exc)
                print(f"  {RED}✗  Failed to launch: {err_msg}{RESET}")
                self.logger.error(f"Launch failed for {phase.name}: {err_msg}")
                results.append(ResumeResult(
                    phase_name=phase.name, phase_prefix=phase.mdp_key,
                    checkpoint_used=cpt_used, command=cmd_str,
                    success=False, error=err_msg,
                ))

            # Phases are serial — only resume first incomplete phase
            # (NVT must finish before NPT, etc.)
            break

        return results




# ─────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
DIM    = "\033[2m"


def _bar(pct: float, width: int = 30) -> str:
    filled = int(width * pct / 100)
    bar    = "█" * filled + "░" * (width - filled)
    color  = GREEN if pct == 100 else (YELLOW if pct > 0 else DIM)
    return f"{color}[{bar}]{RESET} {pct:6.2f}%"


def print_report(report: SimulationReport, verbose: bool = False):
    line = "─" * 70
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  MD SIMULATION CHECKPOINT REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"  {DIM}Type    :{RESET} {report.sim_type.replace('_', ' ').title()}")
    print(f"  {DIM}Dir     :{RESET} {report.sim_dir}")
    print(f"  {DIM}Created :{RESET} {report.generated_at}")
    print(f"  {DIM}Overall :{RESET} {_bar(report.overall_progress_pct)}")
    print(f"{CYAN}{line}{RESET}")

    # ── per-phase ──────────────────────────────────────────
    for phase in report.phases:
        status = (
            f"{GREEN}✔ DONE{RESET}"       if phase.completed
            else f"{YELLOW}⏳ RUNNING{RESET}" if phase.progress_pct > 0
            else f"{DIM}○  PENDING{RESET}"
        )
        print(f"\n  {BOLD}{phase.name}{RESET}  {status}")
        print(f"    Progress : {_bar(phase.progress_pct)}")
        if phase.total_time_ns > 0:
            print(f"    Time     : {phase.current_time_ns:.4f} / {phase.total_time_ns:.4f} ns")
        if phase.wall_time_elapsed:
            print(f"    Elapsed  : {phase.wall_time_elapsed}")
        if phase.estimated_remaining and not phase.completed:
            print(f"    Remaining: {phase.estimated_remaining}")
        if phase.last_checkpoint:
            print(f"    Checkpoint: {phase.last_checkpoint}")
        if verbose and phase.notes:
            for note in phase.notes:
                print(f"    {DIM}ℹ  {note}{RESET}")

    # ── energy stats ───────────────────────────────────────
    es = report.energy_stats
    has_energy = any([
        es.temperature, es.pressure, es.potential_energy,
        es.total_energy, es.density,
    ])
    if has_energy:
        print(f"\n{CYAN}{line}{RESET}")
        print(f"  {BOLD}Last Recorded Energy Statistics{RESET}")
        if es.temperature      is not None: print(f"    Temperature      : {es.temperature:.2f} K")
        if es.pressure         is not None: print(f"    Pressure         : {es.pressure:.2f} bar")
        if es.potential_energy is not None: print(f"    Potential Energy : {es.potential_energy:.2f} kJ/mol")
        if es.total_energy     is not None: print(f"    Total Energy     : {es.total_energy:.2f} kJ/mol")
        if es.density          is not None: print(f"    Density          : {es.density:.2f} kg/m³")

    # ── warnings ───────────────────────────────────────────
    if report.warnings:
        print(f"\n{CYAN}{line}{RESET}")
        print(f"  {BOLD}{YELLOW}Warnings{RESET}")
        for w in report.warnings:
            print(f"    {YELLOW}⚠  {w}{RESET}")

    # ── resume commands ────────────────────────────────────
    if report.resume_commands:
        print(f"\n{CYAN}{line}{RESET}")
        print(f"  {BOLD}Resume Commands{RESET}")
        for cmd in report.resume_commands:
            print()
            for line_ in cmd.split("\n"):
                print(f"    {BLUE}{line_}{RESET}")

    print(f"\n{CYAN}{'═'*70}{RESET}\n")


# ─────────────────────────────────────────────────────────────
# JSON EXPORT
# ─────────────────────────────────────────────────────────────

def export_json(report: SimulationReport, out_path: str):
    data = {
        "sim_type":           report.sim_type,
        "sim_dir":            report.sim_dir,
        "generated_at":       report.generated_at,
        "overall_progress":   report.overall_progress_pct,
        "phases":             [asdict(p) for p in report.phases],
        "energy_stats":       asdict(report.energy_stats),
        "warnings":           report.warnings,
        "resume_commands":    report.resume_commands,
    }
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  Report saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GROMACS MD Simulation Checkpoint Progress Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Protein-in-water simulation
  python md_checkpoint_tracker.py --dir ./sim_protein_water --type protein_water

  # Protein-ligand simulation
  python md_checkpoint_tracker.py --dir ./sim_complex --type protein_ligand --ligand LIG

  # Auto-resume a stopped simulation (Marlowe single-node config)
  python md_checkpoint_tracker.py --dir ./sim --auto-resume --ntmpi 1 --ntomp 16 --gpu-id 0

  # Dry-run: show what would be executed without launching
  python md_checkpoint_tracker.py --dir ./sim --auto-resume --dry-run

  # Watch mode with auto-resume on stall detection (refresh every 2 min)
  python md_checkpoint_tracker.py --dir ./sim --watch --interval 120 --auto-resume

  # Export JSON report
  python md_checkpoint_tracker.py --dir ./sim --json report.json
        """,
    )
    p.add_argument("--dir",         required=True,  help="Path to GROMACS simulation directory")
    p.add_argument("--type",        default="protein_water",
                   choices=["protein_water", "protein_ligand"],
                   help="Simulation type (default: protein_water)")
    p.add_argument("--ligand",      default="LIG",  help="Ligand residue name (default: LIG)")
    p.add_argument("--gmx",         default=None,   help="GROMACS binary (default: auto-detect)")

    # ── compute resources ──────────────────────────────────
    res = p.add_argument_group("Compute resources (used when resuming)")
    res.add_argument("--ntmpi",     type=int, default=1,   help="Number of MPI ranks (default: 1)")
    res.add_argument("--ntomp",     type=int, default=8,   help="OpenMP threads per rank (default: 8)")
    res.add_argument("--gpu-id",    default="0",           help="GPU ID string (default: 0; pass '' to disable)")
    res.add_argument("--no-pin",    action="store_true",   help="Disable CPU pinning")

    # ── modes ──────────────────────────────────────────────
    p.add_argument("--auto-resume", action="store_true",   help="Automatically resume incomplete phases")
    p.add_argument("--dry-run",     action="store_true",   help="With --auto-resume: print command only, do not execute")
    p.add_argument("--watch",       action="store_true",   help="Poll continuously")
    p.add_argument("--interval",    type=int, default=60,  help="Watch interval in seconds (default: 60)")
    p.add_argument("--json",        default=None,          help="Export JSON report to this file")
    p.add_argument("--verbose",     action="store_true",   help="Show per-phase notes")
    return p


def print_resume_summary(results: list[ResumeResult]):
    if not results:
        return
    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"  {BOLD}Resume Summary{RESET}")
    for r in results:
        icon  = f"{GREEN}✔{RESET}" if r.success else f"{RED}✗{RESET}"
        pid_s = f"  PID {r.pid}" if r.pid else ""
        print(f"    {icon}  {r.phase_name}{pid_s}")
        print(f"       Checkpoint : {r.checkpoint_used}")
        if r.error:
            print(f"       {RED}Error      : {r.error}{RESET}")
    print()


def main():
    args = build_parser().parse_args()

    if not os.path.isdir(args.dir):
        print(f"{RED}ERROR: Directory not found: {args.dir}{RESET}", file=sys.stderr)
        sys.exit(1)

    tracker = MDCheckpointTracker(
        sim_dir     = args.dir,
        sim_type    = args.type,
        ligand_name = args.ligand,
        gmx         = args.gmx,
        ntmpi       = args.ntmpi,
        ntomp       = args.ntomp,
        gpu_id      = args.gpu_id,
        pin         = not args.no_pin,
    )

    def _run_once() -> SimulationReport:
        report = tracker.run()
        if args.watch:
            os.system("clear" if os.name != "nt" else "cls")
        print_report(report, verbose=args.verbose)
        if args.json:
            export_json(report, args.json)
        return report

    if args.watch:
        print(f"{CYAN}Watching simulation in {args.dir} (Ctrl-C to stop){RESET}")
        try:
            while True:
                report = _run_once()
                if report.overall_progress_pct >= 100.0:
                    print(f"{GREEN}{BOLD}All phases complete!{RESET}")
                    break

                # Auto-resume on stall if requested
                if args.auto_resume:
                    stalled = any(
                        w for w in report.warnings if "stalled" in w.lower()
                    )
                    if stalled:
                        print(f"\n{YELLOW}Stall detected — attempting auto-resume…{RESET}")
                        results = tracker.auto_resume(dry_run=args.dry_run)
                        print_resume_summary(results)

                print(f"{DIM}Next refresh in {args.interval}s …{RESET}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Monitoring stopped.{RESET}")

    elif args.auto_resume:
        # One-shot: report then resume
        report = _run_once()
        if report.overall_progress_pct < 100.0:
            print(f"\n{CYAN}Auto-resuming incomplete phases…{RESET}")
            results = tracker.auto_resume(dry_run=args.dry_run)
            print_resume_summary(results)
        else:
            print(f"{GREEN}{BOLD}All phases already complete!{RESET}")

    else:
        _run_once()


if __name__ == "__main__":
    main()

