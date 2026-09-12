import os
import yaml
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Trinity Ablation Suite")
    parser.add_argument('--models', nargs='+', default=['resnet18', 'wide_resnet', 'transformer'],
                        help="Models to run (resnet18, wide_resnet, transformer)")
    parser.add_argument('--base_config', default='config.yaml', help="Base config file")
    args = parser.parse_args()

    # Load base config
    with open(args.base_config, 'r') as f:
        base_cfg = yaml.safe_load(f)
    
    # Ablation variants defined in Trinity Paper Plan (Group D)
    ablations = [
        {
            "name": "adam", 
            "opt": "adam", 
            "overrides": {}
        },
        {
            "name": "kfac_always_on", 
            "opt": "trinity", 
            "overrides": {"use_so": True, "force_so_always": True, "use_zo": False, "use_escape": False, "use_gc": True, "use_grafting": True}
        },
        {
            "name": "trinity_full", 
            "opt": "trinity", 
            "overrides": {"use_so": True, "force_so_always": False, "use_zo": True, "use_escape": True, "use_gc": True, "use_grafting": True}
        },
        {
            "name": "trinity_no_escape", 
            "opt": "trinity", 
            "overrides": {"use_so": True, "force_so_always": False, "use_zo": True, "use_escape": False, "use_gc": True, "use_grafting": True}
        },
        {
            "name": "trinity_no_gc", 
            "opt": "trinity", 
            "overrides": {"use_so": True, "force_so_always": False, "use_zo": True, "use_escape": True, "use_gc": False, "use_grafting": True}
        },
        {
            "name": "trinity_no_grafting", 
            "opt": "trinity", 
            "overrides": {"use_so": True, "force_so_always": False, "use_zo": True, "use_escape": True, "use_gc": True, "use_grafting": False}
        },
        {
            "name": "trinity_without_kfac", 
            "opt": "trinity", 
            "overrides": {"use_so": False, "force_so_always": False, "use_zo": True, "use_escape": True, "use_gc": True, "use_grafting": False}
        }
    ]
    
    script_map = {
        'resnet18': 'train_resnet18_dsaa.py',
        'wide_resnet': 'train_wide_resnet_dsaa.py',
        'transformer': 'train_transformer_dsaa.py'
    }
    
    for model_name in args.models:
        model_script = script_map.get(model_name)
        if not model_script or not os.path.exists(model_script):
            print(f"Skipping {model_name}, script {model_script} not found.")
            continue

        for ab in ablations:
            print(f"\n{'='*50}\nRunning {model_name} with ablation: {ab['name']}\n{'='*50}")
            
            # Deep copy base config
            cfg = yaml.safe_load(yaml.dump(base_cfg))
            
            # Apply ablation overrides
            if ab['opt'] == 'trinity':
                for k, v in ab['overrides'].items():
                    cfg['optimizers']['trinity'][k] = v
            
            temp_config = f"config_ablation_{ab['name']}.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(cfg, f)
            
            cmd = ["python", model_script, "--optimizer", ab['opt'], "--config", temp_config, "--output_dir", "ablation"]
            print(f"Command: {' '.join(cmd)}\n")
            
            # Execute command
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running {ab['name']} on {model_name}: {e}")
            
            # Cleanup temp config
            if os.path.exists(temp_config):
                os.remove(temp_config)

if __name__ == "__main__":
    main()
