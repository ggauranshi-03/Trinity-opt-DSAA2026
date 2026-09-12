import os
import yaml
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Trinity Ablation Suite")
    parser.add_argument('--models', nargs='+',
                        default=['resnet18', 'wide_resnet', 'transformer', 'transformer_small'],
                        help="Models to run (resnet18, wide_resnet, transformer, transformer_small)")
    parser.add_argument('--base_config', default='config.yaml', help="Base config file")
    parser.add_argument('--output_dir', default='ablation', help="Output dir for per-run CSVs")
    parser.add_argument('--wandb_project', default=None,
                        help="Literal wandb project name for every run in this ablation sweep "
                             "(passed through as --wandb_project to the training script).")
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
        'transformer': 'train_transformer_dsaa.py',
        'transformer_small': 'train_transformer_small_dsaa.py',
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

            # Apply ablation overrides to the shared trinity block, AND to this model's
            # per-architecture override sub-block if one exists (config.yaml's
            # optimizers.trinity.overrides.<model_name>). Every model now has such a
            # block with its own tuned lr/damping/sensor settings, and build_optimizer
            # applies it AFTER the shared block -- so without this, an ablation like
            # "trinity_no_gc" would silently be clobbered back to whatever use_gc the
            # model's tuned override pins (e.g. transformer_small always forces
            # use_gc=False), making that ablation arm identical to "trinity_full"
            # instead of actually isolating the component. Only the ablation's own keys
            # are touched here -- the model override's tuned lr/damping/sensor values
            # are left alone, which is correct: an ablation should hold everything else
            # fixed and vary only the flag under test.
            if ab['opt'] == 'trinity':
                for k, v in ab['overrides'].items():
                    cfg['optimizers']['trinity'][k] = v
                model_override = cfg['optimizers']['trinity'].get('overrides', {}).get(model_name)
                if model_override is not None:
                    for k, v in ab['overrides'].items():
                        model_override[k] = v

            temp_config = f"config_ablation_{ab['name']}.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(cfg, f)

            cmd = ["python", model_script, "--optimizer", ab['opt'], "--config", temp_config,
                   "--output_dir", args.output_dir]
            # --run_name matters regardless of --wandb_project: 6 of 7 arms share
            # --optimizer trinity, and each training script now uses run_name (falling
            # back to --optimizer) for BOTH the wandb run name and the output CSV
            # filename -- without it, every trinity-variant arm would overwrite the
            # same "..._trinity_..." CSV, leaving only the last arm's data.
            cmd += ["--run_name", ab['name']]
            if args.wandb_project:
                # Same literal project for every ablation arm (as requested) -- they're
                # meant to land side by side in one wandb project.
                cmd += ["--wandb_project", args.wandb_project]
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
