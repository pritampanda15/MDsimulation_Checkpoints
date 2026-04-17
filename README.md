# MD Simulation Checkpoint Progress Tracker

## Phase tracking
Automatically walks through the canonical 4-phase GROMACS pipeline (EM → NVT → NPT → Production MD) for both protein-water and protein-ligand setups. Each phase is checked independently so partial completion is handled cleanly.

## Progress parsing 
Reads nsteps and dt from the .log file to calculate total simulation time, then finds the latest Step / Time entry for current progress. Gives you current_ns / total_ns and a percentage bar.

## Energy statistics 
Scrapes the last recorded Temperature, Pressure, Potential Energy, Total Energy, and Density directly from the log without needing gmx energy (no binary dependency at runtime).

## Smart warnings 
Flags stale checkpoints (>1 hr old, likely stalled job), temperature out of 200–400 K range, and extreme pressure values.
Resume commands auto-generates the correct gmx mdrun -cpi command for any in-progress phase.

```
# Protein in water
python md_checkpoint_tracker.py --dir ./sim_apoprotein --type protein_water

# Protein-ligand (e.g., propofol / GABA-A complex)
python md_checkpoint_tracker.py --dir ./sim_complex --type protein_ligand --ligand PRO

# Watch mode on Marlowe (refresh every 2 min)
python md_checkpoint_tracker.py --dir ./sim --watch --interval 120 --verbose

# Dump JSON for downstream parsing / MDInsight integration
python md_checkpoint_tracker.py --dir ./sim --json checkpoint_report.json

```
