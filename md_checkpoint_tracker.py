#!/usr/bin/env python3
"""
MD Simulation Checkpoint Progress Tracker
==========================================
Tracks and reports progress of GROMACS MD simulations.
Supports both protein-in-water and protein-ligand setups.

Usage:
    python md_checkpoint_tracker.py --dir /path/to/sim --type protein_water
    python md_checkpoint_tracker.py --dir /path/to/sim --type protein_ligand --ligand LIG
    python md_checkpoint_tracker.py --dir /path/to/sim --watch --interval 60

Author : Pritam Kumar Panda
Affil  : Stanford University
"""

import os
import re
import sys
import time
import glob
import argparse
import subprocess
import json
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


def find_checkpoint(sim_dir: str, prefix: str) -> Optional[str]:
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
    ):
        self.sim_dir     = os.path.abspath(sim_dir)
        self.sim_type    = sim_type
        self.ligand_name = ligand_name
        self.gmx         = gmx or gmx_binary()

        template_phases = (
            PROTEIN_LIGAND_PHASES if sim_type == "protein_ligand"
            else PROTEIN_WATER_PHASES
        )
        # Deep-copy so we mutate per-instance state
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
        cpt_path  = find_checkpoint(sim_dir, prefix)

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

    def _build_resume_commands(self) -> list[str]:
        cmds = []
        for phase in self.phases:
            if phase.completed or phase.progress_pct == 0.0:
                continue
            cpt = find_checkpoint(self.sim_dir, phase.mdp_key)
            if cpt:
                cmd = (
                    f"{self.gmx} mdrun -v -deffnm {phase.mdp_key} "
                    f"-cpi {os.path.basename(cpt)} -nt 8 -gpu_id 0"
                )
                cmds.append(f"# Resume {phase.name}\n{cmd}")
        return cmds


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

  # Watch mode (refresh every 2 min)
  python md_checkpoint_tracker.py --dir ./sim --watch --interval 120

  # Export JSON report
  python md_checkpoint_tracker.py --dir ./sim --json report.json
        """,
    )
    p.add_argument("--dir",      required=True,  help="Path to GROMACS simulation directory")
    p.add_argument("--type",     default="protein_water",
                   choices=["protein_water", "protein_ligand"],
                   help="Simulation type (default: protein_water)")
    p.add_argument("--ligand",   default="LIG",  help="Ligand residue name (default: LIG)")
    p.add_argument("--gmx",      default=None,   help="GROMACS binary (default: auto-detect)")
    p.add_argument("--watch",    action="store_true", help="Poll continuously")
    p.add_argument("--interval", type=int, default=60,  help="Watch interval in seconds (default: 60)")
    p.add_argument("--json",     default=None,   help="Export JSON report to this file")
    p.add_argument("--verbose",  action="store_true",   help="Show per-phase notes")
    return p


def main():
    args = build_parser().parse_args()

    if not os.path.isdir(args.dir):
        print(f"{RED}ERROR: Directory not found: {args.dir}{RESET}", file=sys.stderr)
        sys.exit(1)

    tracker = MDCheckpointTracker(
        sim_dir    = args.dir,
        sim_type   = args.type,
        ligand_name= args.ligand,
        gmx        = args.gmx,
    )

    def _run_once():
        report = tracker.run()
        # Clear terminal in watch mode
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
                print(f"{DIM}Next refresh in {args.interval}s …{RESET}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Monitoring stopped.{RESET}")
    else:
        _run_once()


if __name__ == "__main__":
    main()
