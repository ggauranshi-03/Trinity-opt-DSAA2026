import re
import os

log_files = {
    '1': {
        'title': 'Wide-ResNet-101-2',
        'trinity': 'logs/trinity_wideresent_new.log',
        'sgd': 'logs/sgd_wresnet.log',
        'rmsprop': 'logs/rmsprop_wresnet.log',
        'adam': 'logs/adam_wresnet.log',
        'ymin': '0.0001', 'ymax': '4.0'
    },
    '2': {
        'title': 'Vision Transformer',
        'trinity': 'logs/trinity_transformer_new.log',
        'sgd': 'logs/sgd_transformer_new.log',
        'rmsprop': 'logs/rmsprop_transformer_new.log',
        'adam': 'logs/adam_transformer_new.log',
        'ymin': '0.00005', 'ymax': '2.5'
    },
    '3': {
        'title': 'ResNet-18',
        'trinity': 'logs/trinity_resnet_new.log',
        'sgd': 'logs/sgd.log',
        'rmsprop': 'logs/rmsprop.log',
        'adam': 'logs/adam.log',
        'ymin': '0.0005', 'ymax': '3.0'
    }
}

pattern = re.compile(r'Epoch (\d+)/\d+.*?Loss:\s*([0-9.]+)')

def get_coords(filepath):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                coords.append(f"({m.group(1)},{m.group(2)})")
    return " ".join(coords)

with open('latex_snippets.txt', 'w') as out:
    for exp_id, data in log_files.items():
        coords_tr = get_coords(data['trinity'])
        coords_ad = get_coords(data['adam'])
        coords_rm = get_coords(data['rmsprop'])
        coords_sg = get_coords(data['sgd'])
        
        snippet = f"""% --- Experiment {exp_id} ---
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
    \\begin{{axis}}[
      width=0.95\\columnwidth,
      height=5.8cm,
      ymode=log,
      xlabel={{Epoch}},
      ylabel={{Training Loss}},
      xlabel style={{font=\\small, yshift=2pt}},
      ylabel style={{font=\\small, xshift=-2pt}},
      tick label style={{font=\\footnotesize}},
      xmin=1,   xmax=200,
      ymin={data['ymin']}, ymax={data['ymax']},
      xtick={{1,25,50,75,100,125,150,175,200}},
      minor tick num=4,
      grid=none,
      legend pos=south west,
      legend style={{
        font=\\footnotesize,
        fill=white,
        fill opacity=0.95,
        text opacity=1,
        row sep=2pt,
        inner sep=4pt,
        rounded corners=1pt
      }},
      clip=true,
    ]
    \\addplot[line width=1.3pt, color=colorhybrid, mark=none] coordinates {{
      {coords_tr}
    }};
    \\addplot[line width=1.3pt, color=coloradam, mark=none] coordinates {{
      {coords_ad}
    }};
    \\addplot[line width=1.3pt, color=colorrmsprop, mark=none] coordinates {{
      {coords_rm}
    }};
    \\addplot[line width=1.3pt, color=colorsgd, mark=none] coordinates {{
      {coords_sg}
    }};
    \\legend{{Trinity (Proposed), Adam, RMSProp, SGD}}
    \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{Experiment {exp_id} ({data['title']}) convergence: Training Loss trajectories over 200 epochs.}}
  \\label{{fig:exp{exp_id}}}
\\end{{figure}}
"""
        out.write(snippet + "\n")
