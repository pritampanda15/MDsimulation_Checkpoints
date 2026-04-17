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

## How GROMACS Resume Works (what the script wraps)

When `mdrun` is stopped (SLURM walltime, `scancel`, crash, Ctrl-C), GROMACS writes two checkpoint files before dying:

| File | Purpose |
|---|---|
| `md.cpt` | Last checkpoint (coordinates, velocities, RNG state) |
| `md_prev.cpt` | The one before — safety backup |

The entire simulation state is encoded in `.cpt` — so resuming is just pointing `mdrun` at it via `-cpi`.

---

## What the Script Does

**Step 1 — Detect incomplete phases**
```python
def _build_resume_commands(self) -> list[str]:
    for phase in self.phases:
        if phase.completed or phase.progress_pct == 0.0:
            continue   # skip done and not-started phases
```
It skips phases that are 100% done or haven't started. Only phases with `0 < progress < 100` get a resume command.

**Step 2 — Find the latest `.cpt`**
```python
def find_checkpoint(sim_dir, prefix):
    candidates = sorted(
        glob.glob(f"{sim_dir}/{prefix}*.cpt"),
        key=os.path.getmtime,   # sort by modification time
        reverse=True
    )
    return candidates[0]   # most recent wins
```
It globs `md*.cpt`, `nvt*.cpt` etc. and picks the **newest by mtime** — so `md.cpt` is preferred over `md_prev.cpt` unless the former is corrupted.

**Step 3 — Build the exact resume command**
```python
cmd = (
    f"{self.gmx} mdrun -v -deffnm {phase.mdp_key} "
    f"-cpi {os.path.basename(cpt)} -nt 8 -gpu_id 0"
)
```

The critical flags:

| Flag | What it does |
|---|---|
| `-deffnm md` | Reuse same output filenames — appends to existing `.xtc`/`.edr` |
| `-cpi md.cpt` | **Resume from checkpoint** — reads coordinates, velocities, step counter |
| `-nt 8` | Thread count (adjust to your node) |
| `-gpu_id 0` | GPU assignment |

GROMACS reads the `.tpr` (which still has the full run parameters) + `.cpt` (which has the current state) and continues **from the exact step it stopped**, not from the beginning.

---

## What You Should Tweak for Marlowe

The script generates a generic resume command. For your SLURM setup on Marlowe, you'd want:

```bash
# Marlowe-style (single node, avoid MPI issues as you configured)
gmx mdrun -v -deffnm md \
    -cpi md.cpt \
    -ntmpi 1 \
    -ntomp 16 \
    -gpu_id 0 \
    -pin on
```

Or wrap it in a SLURM script the script doesn't generate that automatically, but you can pipe the output to a `.sh` file:

```bash
python md_checkpoint_tracker.py --dir ./sim --json report.json
# then parse report.json → resume_commands → write a new SBATCH script
```

# One-shot resume (single-node config)
```bash
python md_checkpoint_tracker.py --dir ./sim_complex \
    --type protein_ligand --auto-resume \
    --ntmpi 1 --ntomp 16 --gpu-id 0

# Dry run first to sanity-check the command
python md_checkpoint_tracker.py --dir ./sim --auto-resume --dry-run

# Watch + auto-resume on stall (polls every 5 min)
python md_checkpoint_tracker.py --dir ./sim \
    --auto-resume --watch --interval 300
```
---

## One Gotcha

If the `.cpt` is **corrupted** (node crash mid-write), fall back to `md_prev.cpt`:

```bash
gmx mdrun -v -deffnm md -cpi md_prev.cpt -nt 8
```

The script currently picks newest-by-mtime, so you'd do this manually. A future improvement would be to run `gmx check -f md.cpt` to validate before recommending it.
```
