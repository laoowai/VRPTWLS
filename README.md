# VRPTWLS

Hybrid constructive heuristic for the Vehicle Routing Problem with Time Windows (VRPTW).  
The solver combines savings-based route construction with a local-search post optimisation
phase and produces multiple artefacts (text logs, diagnostics, and Matplotlib plots) for each
run.

## Features
- Eight seed-selection strategies with savings-based insertions.
- Local-search refinements (relocate and 2-opt) with feasibility checks.
- Automatic creation of summaries, benchmark-style tables, and diagnostic reports.
- Optional Matplotlib visualisations of the resulting routes.

## Requirements
- Python 3.9+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [psutil](https://psutil.readthedocs.io/)

Install the Python dependencies with:

```bash
pip install -r requirements.txt  # or: pip install numpy matplotlib psutil
```

## Input data
The program expects Solomon-style VRPTW benchmark files (e.g. `C101.txt`). Each file should
contain the depot and customer rows in the standard tabular format. Point the program at the
directory that contains these files when prompted.

## Running the solver
Execute the main script directly with Python:

```bash
python VRPTWLS.py
```

You will be asked for:
1. **Instance name** – supply values such as `c101` (case-insensitive) or type `all` to process
every predefined Solomon instance group.
2. **Directory path** – the folder where the instance `.txt` files live.

### Single instance example
```
Enter instance name (e.g., c101) or 'all': c101
Enter directory path: /path/to/solomon
```
The program loads `/path/to/solomon/C101.txt`, runs the heuristic, and prints the pre/post
optimisation vehicle count and total distance to the console.

### Batch processing (`all`)
When `all` is entered, the script iterates over every Solomon group (`R1`, `R2`, `C1`, `C2`,
`RC1`, `RC2`). For each instance it reports the before/after metrics and aggregates the results
into summary tables.

## Output artefacts
Runs create (or reuse) several folders in the project root:

| Folder | Contents |
| --- | --- |
| `log_simple_route/` | Concise text logs containing before/after metrics and the customer sequence per route. |
| `log_detailed_route/` | Detailed reports with hardware info, per-route statistics, and time-window slack. |
| `route_matp/` | PNG plots of the computed routes. |
| `logs_results_common/` | A combined summary of vehicles and distance over all processed instances. |
| `logs_results_tables/` | Benchmark-style tables that compare averages before/after local search. |
| `diagnostics/` | Enhanced diagnostics per instance and an aggregate diagnostic summary when processing all instances. |

Example snippet from a simple log (`log_simple_route/C101_simple.txt`):
```
Best Strategy: 2  Total Distance: 832.417
Number of vehicles: 10       Loaded customers: 100       Total distance: 832.417
Route: 0 90 Vehicle: 0 Distance: 81.5020 Customers: 11: 0 57 12 64 90 95 63 66 46 61 21 0
...
Local Search optimization process completed
Number of vehicles: 9       Loaded customers: 100       Total distance: 805.233
Route: 0 96 Vehicle: 0 Distance: 78.1145 Customers: 12: 0 57 64 90 95 63 66 46 61 21 87 12 0
Time spent: 2450.17 milliseconds
```

> **Tip:** To enable verbose strategy comparisons during construction, instantiate
> `HybridSolver` with `enable_logging=True` (see the `process_all_instances` function for a usage
> example).

## Reproducing benchmark tables
After processing all instances, the solver writes two comparison tables mimicking literature
surveys to `logs_results_tables/`. These files show averaged vehicle counts and distances for
both the construction and local-search phases. They can be used to compare multiple runs or
baseline results.

## License
This project is released under the terms of the [MIT License](LICENSE).
