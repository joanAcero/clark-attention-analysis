# Running the experiments with Slurm

Scripts for running [REPRODUCE.md](../REPRODUCE.md) on a Slurm cluster. Paths are set in
`config.sh`; edit it first. Submit every job **from the repository root** and pass your
project with `-A` (list your projects with `Sproject`).

| Step | Command | Where | What it does |
| --- | --- | --- | --- |
| 0 | `bash slurm/prepare_data.sh` | login node | downloads bert-base-uncased and UD EWT/PUD, converts them to JSON, and recovers the inputs of Clark's released maps |
| 1 | `sbatch -A <project> slurm/00_tests.sh` | cpu | unit tests, including the comparison with Kobayashi et al.'s implementation |
| 2 | `sbatch -A <project> slurm/01_extract_wiki.sh` | gpu | Experiment A: attention + norms on the 1000 Wikipedia segments, reproduction check against the released maps, JS distances between heads |
| 3 | `sbatch -A <project> slurm/02_extract_syntax.sh` | gpu | Experiments B and C: word-level maps for EWT train/dev and PUD, plus the random-initialization control |
| 4 | `sbatch -A <project> slurm/03_notebooks.sh all` | cpu | runs the notebooks headless (`general`, `ewt`, `pud`, `ewt-ext`, `pud-ext` or `all`); the executed notebooks with figures go to `$RESULTS` |

Logs are written to `logs/<job-name>_<jobid>.out` and `.err`. Check the queue with `squeue -u $USER`.

Before step 0, download Clark et al.'s data from the Google Drive link in the main README into
`$DATA/clark/`, i.e. `$DATA/clark/unlabeled_attn.pkl` and optionally `$DATA/clark/glove/`.
Without it, step 2 cannot run, but steps 3 and 4 (`ewt`, `pud`) can.

Steps 2 and 3 are independent and can run at the same time. To start step 4 automatically
after both finish:
```bash
A=$(sbatch -A <project> --parsable slurm/01_extract_wiki.sh)
B=$(sbatch -A <project> --parsable slurm/02_extract_syntax.sh)
sbatch -A <project> --dependency=afterok:$A:$B slurm/03_notebooks.sh all
```

**Check after step 2.** The `compare` block in `logs/extract_wiki_<jobid>.out` should show
differences of about 1e-3 or less and an argmax agreement close to 1. If the differences
are much larger, rerun with the other segment-id convention:
```bash
sbatch -A <project> --export=ALL,SEGMENT_IDS=pair slurm/01_extract_wiki.sh
```
