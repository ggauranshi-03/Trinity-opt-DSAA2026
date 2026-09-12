import re
import os

log_files = {
    'Exp1_WideResNet': {
        'Trinity': 'logs/trinity_wideresent_new.log',
        'SGD': 'logs/sgd_wresnet.log',
        'RMSProp': 'logs/rmsprop_wresnet.log',
        'Adam': 'logs/adam_wresnet.log'
    }
}

def parse_log(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    losses = []
    val_accs = []
    
    pattern = re.compile(r'Epoch (\d+)/\d+.*?Loss:\s*([0-9.]+).*?Val Acc:\s*([0-9.]+)%')
    
    for line in lines:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            val_accs.append(float(m.group(3)))
            
    if not epochs:
        return None
        
    return {
        'final_loss': losses[-1],
        'peak_acc': max(val_accs),
        'final_acc': val_accs[-1],
    }

for exp_name, optimizers in log_files.items():
    print(f"=== {exp_name} ===")
    for opt_name, log_file in optimizers.items():
        data = parse_log(log_file)
        if data:
            print(f"{opt_name:<10} | Peak Acc: {data['peak_acc']:>6.2f}% | Final Acc: {data['final_acc']:>6.2f}% | Final Loss: {data['final_loss']:>8.4f}")
        else:
            print(f"{opt_name:<10} | Log file not found or unparsable: {log_file}")
