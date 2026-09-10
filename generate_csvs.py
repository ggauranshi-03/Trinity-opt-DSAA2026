import re
import os
import glob

def parse_log(filepath, output_csv):
    epoch_val_acc = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'Epoch (\d+)/200 - Loss: [0-9.]+, Val Acc: ([0-9.]+)%', line)
            if match:
                epoch = match.group(1)
                val_acc = match.group(2)
                epoch_val_acc.append((epoch, val_acc))
    
    with open(output_csv, 'w') as f:
        f.write("epoch,val_acc\n")
        for epoch, val_acc in epoch_val_acc:
            f.write(f"{epoch},{val_acc}\n")
    return max([float(x[1]) for x in epoch_val_acc]) if epoch_val_acc else 0.0

mapping = {
    'logs/trinity_wresnet.log': 'exp1_hybrid.csv',
    'logs/adam_wresnet.log': 'exp1_adam.csv',
    'logs/rmsprop_wresnet.log': 'exp1_rmsprop.csv',
    'logs/sgd_wresnet.log': 'exp1_sgd.csv',
    
    'logs/trinity.log': 'exp3_hybrid.csv',
    'logs/adam.log': 'exp3_adam.csv',
    'logs/rmsprop.log': 'exp3_rmsprop.csv',
    'logs/sgd.log': 'exp3_sgd.csv'
}

for log_path, csv_path in mapping.items():
    if os.path.exists(log_path):
        max_acc = parse_log(log_path, csv_path)
        print(f"Generated {csv_path} from {log_path} - Max Acc: {max_acc}%")
    else:
        print(f"File not found: {log_path}")

