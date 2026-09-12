import re
import os

def parse_log(filepath, output_csv):
    """
    Parses exactly 200 epochs from a log file.
    Extracts epoch, loss (training loss), val_acc (validation accuracy %),
    and per-group accuracies (head, med, tail) if available.
    """
    pattern = r'Epoch\s+(\d+)/200\s*-\s*Loss:\s*([0-9.]+),\s*Val Acc:\s*([0-9.]+)%(?:,\s*Head:\s*([0-9.]+)%,\s*Med:\s*([0-9.]+)%,\s*Tail:\s*([0-9.]+)%)?'
    
    records = []
    with open(filepath, 'r') as f:
        content = f.read()
    
    for m in re.finditer(pattern, content):
        epoch = int(m.group(1))
        loss = float(m.group(2))
        val_acc = float(m.group(3))
        head = float(m.group(4)) if m.group(4) is not None else 0.0
        med = float(m.group(5)) if m.group(5) is not None else 0.0
        tail = float(m.group(6)) if m.group(6) is not None else 0.0
        records.append((epoch, loss, val_acc, head, med, tail))
    
    assert len(records) == 200, f"Error: expected 200 epochs in {filepath}, found {len(records)}"
    
    with open(output_csv, 'w') as f:
        f.write("epoch,loss,val_acc,head_acc,med_acc,tail_acc\n")
        for ep, loss, val_acc, head, med, tail in records:
            f.write(f"{ep},{loss:.6f},{val_acc:.2f},{head:.2f},{med:.2f},{tail:.2f}\n")
            
    max_acc = max(r[2] for r in records)
    min_loss = min(r[1] for r in records)
    final_loss = records[-1][1]
    final_acc = records[-1][2]
    return max_acc, final_acc, min_loss, final_loss

# Mapping specified by user:
# ResNet: trinity_resnet_new.log, sgd.log, rmsprop.log, adam.log
# WideResNet: trinity_wideresent_new.log, sgd_wresnet.log, rmsprop_wresnet.log, adam_wresnet.log
# Transformer: trinity_transformer_new.log, sgd_transformer_new.log, rmsprop_transformer_new.log, adam_transformer_new.log
mapping = {
    # Exp 1: Wide-ResNet
    'logs/trinity_wideresent_new.log': 'exp1_hybrid.csv',
    'logs/adam_wresnet.log': 'exp1_adam.csv',
    'logs/rmsprop_wresnet.log': 'exp1_rmsprop.csv',
    'logs/sgd_wresnet.log': 'exp1_sgd.csv',
    
    # Exp 2: Transformer
    'logs/trinity_transformer_new.log': 'exp2_hybrid.csv',
    'logs/adam_transformer_new.log': 'exp2_adam.csv',
    'logs/rmsprop_transformer_new.log': 'exp2_rmsprop.csv',
    'logs/sgd_transformer_new.log': 'exp2_sgd.csv',
    
    # Exp 3: ResNet-18
    'logs/trinity_resnet_new.log': 'exp3_hybrid.csv',
    'logs/adam.log': 'exp3_adam.csv',
    'logs/rmsprop.log': 'exp3_rmsprop.csv',
    'logs/sgd.log': 'exp3_sgd.csv'
}

if __name__ == '__main__':
    print(f"{'Source Log':35s} -> {'CSV File':18s} | {'Peak Acc':8s} {'Final Acc':9s} {'Min Loss':9s} {'Final Loss':10s}")
    print("-" * 90)
    for log_path, csv_path in mapping.items():
        if os.path.exists(log_path):
            max_acc, final_acc, min_loss, final_loss = parse_log(log_path, csv_path)
            print(f"{log_path:35s} -> {csv_path:18s} | {max_acc:6.2f}%   {final_acc:6.2f}%   {min_loss:8.4f}  {final_loss:8.4f}")
        else:
            print(f"File not found: {log_path}")
