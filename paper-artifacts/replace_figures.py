import re

with open("paper-artifacts/conference_101719.tex", "r") as f:
    text = f.read()

# Define the replacement blocks
def get_fig_block(exp_num, title, ymin_loss, ymax_loss, ymin_acc, ymax_acc):
    return f"""\\begin{{figure*}}[!t]
  \\centering
  \\begin{{tikzpicture}}
    % --- LOSS PLOT ---
    \\begin{{axis}}[
      name=lossplot,
      width=0.48\\textwidth,
      height=5.0cm,
      ymode=log,
      xlabel={{Epoch}},
      ylabel={{Training Loss}},
      xlabel style={{font=\\small, yshift=2pt}},
      ylabel style={{font=\\small, xshift=-2pt}},
      tick label style={{font=\\footnotesize}},
      xmin=1,   xmax=200,
      ymin={ymin_loss}, ymax={ymax_loss},
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
    \\addplot[line width=1.3pt, color=colorhybrid, mark=none] table [x=epoch, y=loss, col sep=comma] {{CSVs_clean/exp{exp_num}_trinity.csv}};
    \\addplot[line width=1.3pt, color=coloradam, mark=none] table [x=epoch, y=loss, col sep=comma] {{CSVs_clean/exp{exp_num}_adam.csv}};
    \\addplot[line width=1.3pt, color=colorrmsprop, mark=none] table [x=epoch, y=loss, col sep=comma] {{CSVs_clean/exp{exp_num}_rmsprop.csv}};
    \\addplot[line width=1.3pt, color=colorsgd, mark=none] table [x=epoch, y=loss, col sep=comma] {{CSVs_clean/exp{exp_num}_sgd.csv}};
    
    \\legend{{Trinity (Proposed), Adam, RMSProp, SGD}}
    \\end{{axis}}

    % --- ACCURACY PLOT ---
    \\begin{{axis}}[
      at={{(lossplot.east)}},
      anchor=west,
      xshift=1.0cm,
      width=0.48\\textwidth,
      height=5.0cm,
      xlabel={{Epoch}},
      ylabel={{Validation Accuracy (\\%)}},
      xlabel style={{font=\\small, yshift=2pt}},
      ylabel style={{font=\\small, xshift=-2pt}},
      tick label style={{font=\\footnotesize}},
      xmin=1,   xmax=200,
      ymin={ymin_acc}, ymax={ymax_acc},
      xtick={{1,25,50,75,100,125,150,175,200}},
      grid=both,
      legend pos=south east,
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
    \\addplot[line width=1.3pt, color=colorhybrid, mark=none] table [x=epoch, y=val_acc, col sep=comma] {{CSVs_clean/exp{exp_num}_trinity.csv}};
    \\addplot[line width=1.3pt, color=coloradam, mark=none] table [x=epoch, y=val_acc, col sep=comma] {{CSVs_clean/exp{exp_num}_adam.csv}};
    \\addplot[line width=1.3pt, color=colorrmsprop, mark=none] table [x=epoch, y=val_acc, col sep=comma] {{CSVs_clean/exp{exp_num}_rmsprop.csv}};
    \\addplot[line width=1.3pt, color=colorsgd, mark=none] table [x=epoch, y=val_acc, col sep=comma] {{CSVs_clean/exp{exp_num}_sgd.csv}};
    \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{Experiment {exp_num} ({title}) convergence: Training Loss (left) and Validation Accuracy (right).}}
  \\label{{fig:exp{exp_num}}}
\\end{{figure*}}"""

fig1 = get_fig_block(1, "Wide-ResNet-101-2", "0.0001", "4.0", "10", "70")
fig2 = get_fig_block(2, "Vision Transformer", "0.00005", "2.5", "10", "60")
fig3 = get_fig_block(3, "ResNet-18", "0.0005", "3.0", "10", "70")

pattern_fig1 = re.compile(r'\\begin\{figure\}\[t\].*?\\label\{fig:exp1\}\s*\\end\{figure\}', re.DOTALL)
pattern_fig2 = re.compile(r'\\begin\{figure\}\[t\].*?\\label\{fig:exp2\}\s*\\end\{figure\}', re.DOTALL)
pattern_fig3 = re.compile(r'\\begin\{figure\}\[t\].*?\\label\{fig:exp3\}\s*\\end\{figure\}', re.DOTALL)

text = pattern_fig1.sub(lambda m: fig1, text)
text = pattern_fig2.sub(lambda m: fig2, text)
text = pattern_fig3.sub(lambda m: fig3, text)

with open("paper-artifacts/conference_101719.tex", "w") as f:
    f.write(text)

print("Successfully replaced all 3 figure blocks.")
