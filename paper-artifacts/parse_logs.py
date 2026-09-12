import re
import os
import glob

log_files = {
    'Exp1_WideResNet': {
        'Trinity': 'logs/trinity_wideresnet_new.log',
        'SGD': 'logs/sgd_wideresnet.log',
        'RMSProp': 'logs/rmsprop_wresnet.log',
        'Adam': 'logs/adam_wresnet.log'
    },
    'Exp2_Transformer': {
        'Trinity': 'logs/trinity_transformer_new.log',
        'SGD': 'logs/sgd_transformer_new.log',
        'RMSProp': 'logs/rmsprop_transformer_new.log',
        'Adam': 'logs/adam_transformer_new.log'
    },
    'Exp3_ResNet': {
        'Trinity': 'logs/trinity_resnet_new.log',
        'SGD': 'logs/sgd.log',
        'RMSProp': 'logs/rmsprop.log',
        'Adam': 'logs/adam.log'
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
    time_taken = None
    
    # regex for epoch line: Epoch X/200 - Loss: Y, Val Acc: Z%
    # or similar
    pattern = re.compile(r'Epoch (\d+)/\d+.*?Loss:\s*([0-9.]+).*?Val Acc:\s*([0-9.]+)%')
    time_pattern = re.compile(r'Total training time:\s*([0-9.]+)\s*s')
    
    for line in lines:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            val_accs.append(float(m.group(3)))
        
        m_time = time_pattern.search(line)
        if m_time:
            time_taken = float(m_time.group(1))
            
    if not epochs:
        return None
        
    return {
        'epochs': epochs,
        'losses': losses,
        'val_accs': val_accs,
        'final_loss': losses[-1],
        'peak_acc': max(val_accs),
        'final_acc': val_accs[-1],
        'time': time_taken
    }

for exp_name, optimizers in log_files.items():
    print(f"=== {exp_name} ===")
    for opt_name, log_file in optimizers.items():
        data = parse_log(log_file)
        if data:
            print(f"{opt_name:<10} | Peak Acc: {data['peak_acc']:>6.2f}% | Final Acc: {data['final_acc']:>6.2f}% | Final Loss: {data['final_loss']:>8.4f} | Time: {data['time']} | Total Epochs: {len(data['epochs'])}")
        else:
            print(f"{opt_name:<10} | Log file not found or unparsable: {log_file}")
    print()

