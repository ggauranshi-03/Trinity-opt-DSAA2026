"""Combine per-worker Optuna sqlite studies into one ranked report. Each worker writes
its own tune_<prefix>_worker_<id>.db (see tune_trinity_transformer_small.py / tune_trinity_cnn.py
for why: concurrent writers to one sqlite file crash Optuna internally)."""
import argparse
import glob
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument('--glob', default='tune_worker_*.db',
                 help="File glob for this run's worker dbs, e.g. 'tune_resnet18_worker_*.db'")
ap.add_argument('--study_name', default='trinity_transformer_small')
args = ap.parse_args()

all_trials = []
for path in sorted(glob.glob(args.glob)):
    study = optuna.load_study(study_name=args.study_name, storage=f'sqlite:///{path}')
    for t in study.trials:
        if t.state.name == 'COMPLETE':
            all_trials.append((path, t))

print(f"total complete trials across all workers: {len(all_trials)}")
all_trials.sort(key=lambda pt: -pt[1].value)
print()
print("Ranked (best first):")
for path, t in all_trials:
    ua = t.user_attrs
    print(f"  val_acc={t.value:.2f}  n_so={ua.get('n_so')}/{ua.get('n_blocks')}  "
          f"final_val_acc={ua.get('final_val_acc'):.2f}  [{path}]  params={t.params}")

if all_trials:
    best_path, best = all_trials[0]
    print()
    print("BEST:", best.params, f"(from {best_path}, val_acc={best.value:.2f})")
