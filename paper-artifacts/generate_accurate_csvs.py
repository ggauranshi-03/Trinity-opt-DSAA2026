import re
import os
import csv

log_files = {
    'exp1': {
        'trinity': 'logs/trinity_wideresent_new.log',
        'sgd': 'logs/sgd_wresnet.log',
        'rmsprop': 'logs/rmsprop_wresnet.log',
        'adam': 'logs/adam_wresnet.log'
    },
    'exp2': {
        'trinity': 'logs/trinity_transformer_new.log',
        'sgd': 'logs/sgd_transformer_new.log',
        'rmsprop': 'logs/rmsprop_transformer_new.log',
        'adam': 'logs/adam_transformer_new.log'
    },
    'exp3': {
        'trinity': 'logs/trinity_resnet_new.log',
        'sgd': 'logs/sgd.log',
        'rmsprop': 'logs/rmsprop.log',
        'adam': 'logs/adam.log'
    }
}

os.makedirs('paper-artifacts/CSVs_clean', exist_ok=True)

pattern = re.compile(r'Epoch (\d+)/\d+.*?Loss:\s*([0-9.]+).*?Val Acc:\s*([0-9.]+)%')
time_pattern = re.compile(r'Total training time:\s*([0-9.]+)\s*s')

metrics = {}

for exp_id, optimizers in log_files.items():
    metrics[exp_id] = {}
    for opt_id, file_path in optimizers.items():
        if not os.path.exists(file_path):
            print(f"Error: {file_path} does not exist.")
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        csv_path = f"paper-artifacts/CSVs_clean/{exp_id}_{opt_id}.csv"
        
        epochs = []
        losses = []
        val_accs = []
        time_taken = None
        
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['epoch', 'loss', 'val_acc'])
            
            for line in lines:
                m = pattern.search(line)
                if m:
                    ep = int(m.group(1))
                    ls = float(m.group(2))
                    va = float(m.group(3))
                    epochs.append(ep)
                    losses.append(ls)
                    val_accs.append(va)
                    writer.writerow([ep, ls, va])
                
                m_time = time_pattern.search(line)
                if m_time:
                    time_taken = float(m_time.group(1))
                    
        if epochs:
            metrics[exp_id][opt_id] = {
                'peak_acc': max(val_accs),
                'final_acc': val_accs[-1],
                'final_loss': losses[-1],
                'time': time_taken
            }
            print(f"Written {csv_path} with {len(epochs)} epochs.")

# Now let's print the metrics table to verify against the .tex file
print("\n--- METRICS EXTRACTED FROM LOGS ---")
for exp_id, opts in metrics.items():
    print(f"--- {exp_id.upper()} ---")
    for opt_id, data in opts.items():
        print(f"{opt_id:10}: Peak={data['peak_acc']:.2f}%, Final={data['final_acc']:.2f}%, Loss={data['final_loss']:.4f}, Time={data['time']}")

