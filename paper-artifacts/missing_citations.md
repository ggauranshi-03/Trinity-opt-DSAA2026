# Comprehensive Citation Guide for Trinity Paper (`conference_101719.tex`)

This document provides a complete audit of all missing and recommended citations in `conference_101719.tex`. It details:
1. **The exact location in the text** where each citation should be placed (with line numbers and context).
2. **The exact replacement text** with LaTeX `\cite{...}` tags.
3. **The complete `\bibitem` entries** formatted for the IEEE conference reference section.
4. **The BibTeX entries** for authors using a `.bib` file.
5. **Audited status of existing bibliography entries** (including uncited items).

---

## 1. Executive Summary of Citations

| # | Category | Citation Key | Title / Topic | Primary Authors & Year | Missing Status in LaTeX |
|---|---|---|---|---|---|
| 1 | Optimizer | `tieleman2012rmsprop` | RMSProp Optimizer | Tieleman & Hinton, 2012 | In text as plain text, no `\cite` |
| 2 | Second-Order | `grosse2016kronecker` | KFC (Convolutional K-FAC) | Grosse & Martens, 2016 | Named in text (line 106), no `\cite` |
| 3 | Second-Order | `gupta2018shampoo` | Shampoo Optimizer | Gupta et al., 2018 | Typed as `(Gupta et al. 2018)`, no `\cite` |
| 4 | Second-Order | `yao2021adahessian` | AdaHessian Optimizer | Yao et al., 2021 | Named in text (line 106), no `\cite` |
| 5 | Second-Order | `vyas2024soap` | SOAP Optimizer | Vyas et al., 2024 | Typed as `(Vyas et al. 2024)`, no `\cite` |
| 6 | Second-Order | `jordan2024muon` | Muon Optimizer | Jordan et al., 2024 | Named in text (line 106), no `\cite` |
| 7 | Optimization | `agarwal2020disentangling` | Step Magnitude Grafting | Agarwal et al., 2020 | Typed as `(Agarwal et al.)`, no `\cite` |
| 8 | Zeroth-Order | `malladi2023mezo` | MeZO Optimizer | Malladi et al., 2023 | Typed as `(Malladi et al. 2023)`, no `\cite` |
| 9 | Zeroth-Order | `foret2020sharpness` | Sharpness-Aware Minimization (SAM) | Foret et al., 2021 | Typed as `(Foret et al. 2021)`, no `\cite` |
| 10 | Saddle Escape | `ge2015escaping` | Escaping Saddle Points (Perturbed GD) | Ge et al., 2015 | Typed as `(Ge et al. 2015)`, no `\cite` |
| 11 | Saddle Escape | `jin2017escape` | Efficient Saddle Escape | Jin et al., 2017 | Typed as `(Jin et al. 2017)`, no `\cite` |
| 12 | Second-Order | `marquardt1963algorithm` | Levenberg-Marquardt Algorithm | Marquardt, 1963 / Levenberg, 1944 | Named in text (line 112), no `\cite` |
| 13 | Architecture | `he2016deep` | ResNet-18 (Residual Networks) | He et al., 2016 | Central baseline (Exp. 3), no `\cite` |
| 14 | Architecture | `zagoruyko2016wide` | Wide-ResNet-101-2 | Zagoruyko & Komodakis, 2016 | Central baseline (Exp. 1), no `\cite` |
| 15 | Architecture | `dosovitskiy2020image` | Vision Transformer (ViT) | Dosovitskiy et al., 2021 | Central baseline (Exp. 2), no `\cite` |
| 16 | Architecture | `vaswani2017attention` | Self-Attention / Transformer | Vaswani et al., 2017 | Foundational to ViT, no `\cite` |
| 17 | Dataset | `krizhevsky2009learning` | CIFAR-10 Dataset | Krizhevsky & Hinton, 2009 | Experimental dataset, no `\cite` |
| 18 | Imbalance | `cui2019class` | Class-Balanced Loss (Long-Tailed) | Cui et al., 2019 | Explains $\text{IF}=0.01$ benchmark, no `\cite` |
| 19 | Imbalance | `cao2019learning` | LDAM-Loss / Long-Tailed CIFAR | Cao et al., 2019 | Standard benchmark protocol, no `\cite` |
| 20 | Scheduler | `loshchilov2016sgdr` | CosineAnnealingLR (SGDR) | Loshchilov & Hutter, 2017 | LR policy used across all runs, no `\cite` |
| 21 | Tuning | `akiba2019optuna` | Optuna Hyperparameter Optimization | Akiba et al., 2019 | Used for hyperparameter search, no `\cite` |
| 22 | Framework | `paszke2019pytorch` | PyTorch Library | Paszke et al., 2019 | Mentioned in Section IV, no `\cite` |

---

## 2. Where and How to Cite in `conference_101719.tex`

Below is the precise mapping showing the current text, the file location, and the recommended replacement containing the LaTeX citation.

### Location 1: Section II-A (First-Order Adaptive Methods)
* **File & Line**: `conference_101719.tex`, Line 103
* **Current Text**:
  ```latex
  To address sensitivity to hyperparameter tuning and ill-conditioned landscapes, adaptive first-order methods like Adam \cite{kingma2014adam} and RMSProp were introduced.
  ```
* **Recommended Replacement**:
  ```latex
  To address sensitivity to hyperparameter tuning and ill-conditioned landscapes, adaptive first-order methods like Adam \cite{kingma2014adam} and RMSProp \cite{tieleman2012rmsprop} were introduced.
  ```

---

### Location 2: Section II-B (Second-Order Preconditioning and Grafting)
* **File & Line**: `conference_101719.tex`, Line 106
* **Current Text**:
  ```latex
  To achieve tractable scaling, Martens and Grosse \cite{martens2015optimizing} proposed Kronecker-Factored Approximate Curvature (K-FAC) for linear layers, later extended to convolutional layers (KFC) by Grosse and Martens. Extensions of K-FAC have been successfully applied to massive architectures \cite{osawa2019large}. Other recent preconditioning approaches include Shampoo (Gupta et al. 2018), AdaHessian, SOAP (Vyas et al. 2024), and Muon. To safely combine the directional benefits of second-order steps with the robust step sizes of first-order methods, grafting (Agarwal et al.) has emerged as a crucial technique.
  ```
* **Recommended Replacement**:
  ```latex
  To achieve tractable scaling, Martens and Grosse \cite{martens2015optimizing} proposed Kronecker-Factored Approximate Curvature (K-FAC) for linear layers, later extended to convolutional layers (KFC) by Grosse and Martens \cite{grosse2016kronecker}. Extensions of K-FAC have been successfully applied to massive architectures \cite{osawa2019large}. Other recent preconditioning approaches include Shampoo \cite{gupta2018shampoo}, AdaHessian \cite{yao2021adahessian}, SOAP \cite{vyas2024soap}, and Muon \cite{jordan2024muon}. To safely combine the directional benefits of second-order steps with the robust step sizes of first-order methods, grafting \cite{agarwal2020disentangling} has emerged as a crucial technique. In addition, multi-step heuristics such as Lookahead \cite{zhang2019lookahead} explore complementary outer-loop step stabilization.
  ```

---

### Location 3: Section II-C (Zeroth-Order Optimization and Perturbation)
* **File & Line**: `conference_101719.tex`, Line 109
* **Current Text**:
  ```latex
  Recently, ZO principles have been integrated into deep learning as landscape-aware regularizers or memory-efficient optimizers, such as MeZO (Malladi et al. 2023). Crucially, Sharpness-Aware Minimization (SAM) (Foret et al. 2021) established the motivation of escaping sharp minima via an extra forward/backward pass. Additionally, perturbed gradient descent (Ge et al. 2015, Jin et al. 2017) demonstrated that injecting random noise allows the optimizer to efficiently escape saddle points.
  ```
* **Recommended Replacement**:
  ```latex
  Recently, ZO principles have been integrated into deep learning as landscape-aware regularizers or memory-efficient optimizers, such as MeZO \cite{malladi2023mezo}. Crucially, Sharpness-Aware Minimization (SAM) \cite{foret2020sharpness} established the motivation of escaping sharp minima via an extra forward/backward pass. Additionally, perturbed gradient descent \cite{ge2015escaping, jin2017escape} demonstrated that injecting random noise allows the optimizer to efficiently escape saddle points.
  ```

---

### Location 4: Section II-D (Adaptive Curvature Use)
* **File & Line**: `conference_101719.tex`, Line 112
* **Current Text**:
  ```latex
  The classical ancestor of curvature-driven adaptation is the Levenberg-Marquardt (LM) gain ratio, which adapts damping based on the agreement between actual and predicted loss reduction.
  ```
* **Recommended Replacement**:
  ```latex
  The classical ancestor of curvature-driven adaptation is the Levenberg-Marquardt (LM) gain ratio \cite{marquardt1963algorithm}, which adapts damping based on the agreement between actual and predicted loss reduction.
  ```

---

### Location 5: Section IV-A (Dataset and Imbalance Configuration)
* **File & Line**: `conference_101719.tex`, Lines 947–952
* **Current Text**:
  ```latex
  To ensure the reproducibility of our claims, the complete PyTorch implementation of the Trinity framework...
  For all evaluations, we utilized a challenging Long-Tailed version of the CIFAR-10 dataset to simulate real-world label imbalances and out-of-distribution tracking performance. We applied an architectural imbalance factor of 0.01, creating a severe 100:1 ratio between the dominant majority and rare minority classes.
  ```
* **Recommended Replacement**:
  ```latex
  To ensure the reproducibility of our claims, the complete PyTorch \cite{paszke2019pytorch} implementation of the Trinity framework...
  For all evaluations, we utilized a challenging Long-Tailed version of the CIFAR-10 dataset \cite{krizhevsky2009learning} following standard class-imbalance benchmarks \cite{cui2019class, cao2019learning, kunstner2024heavy} to simulate real-world label imbalances and out-of-distribution tracking performance. We applied an architectural imbalance factor of 0.01, creating a severe 100:1 ratio between the dominant majority and rare minority classes.
  ```

---

### Location 6: Section IV-B (Architectural Configurations)
* **File & Line**: `conference_101719.tex`, Lines 963–970
* **Current Text**:
  ```latex
  \item \textbf{Experiment 1 (Wide Residual Network):} We evaluate a Wide-ResNet-101-2 architecture containing approximately $126.9\times 10^6$ parameters...
  \item \textbf{Experiment 2 (Vision Transformer):} We configure a custom transformer comprising approximately $86\times 10^6$ parameters...
  \item \textbf{Experiment 3 (ResNet-18):} We instantiate a compact ResNet-18 model containing approximately $11\times 10^6$ parameters...
  ...learning rate trajectories are modulated across training epochs using a \texttt{CosineAnnealingLR} schedule policy.
  ```
* **Recommended Replacement**:
  ```latex
  \item \textbf{Experiment 1 (Wide Residual Network):} We evaluate a Wide-ResNet-101-2 architecture \cite{zagoruyko2016wide} containing approximately $126.9\times 10^6$ parameters...
  \item \textbf{Experiment 2 (Vision Transformer):} We configure a custom Vision Transformer \cite{dosovitskiy2020image, vaswani2017attention} comprising approximately $86\times 10^6$ parameters...
  \item \textbf{Experiment 3 (ResNet-18):} We instantiate a compact ResNet-18 model \cite{he2016deep} containing approximately $11\times 10^6$ parameters...
  ...learning rate trajectories are modulated across training epochs using a \texttt{CosineAnnealingLR} schedule policy \cite{loshchilov2016sgdr}.
  ```

---

### Location 7: Section IV-C (Hyperparameter Tuning)
* **File & Line**: `conference_101719.tex`, Line 979
* **Current Text**:
  ```latex
  We performed rigorous hyperparameter optimization via the Optuna tuning framework over continuous configuration spaces to isolate peak baseline performance.
  ```
* **Recommended Replacement**:
  ```latex
  We performed rigorous hyperparameter optimization via the Optuna tuning framework \cite{akiba2019optuna} over continuous configuration spaces to isolate peak baseline performance.
  ```

---

## 3. Ready-to-Paste `\bibitem` References (for `conference_101719.tex`)

You can directly paste this consolidated list inside `\begin{thebibliography}{25}` at the end of `conference_101719.tex`:

```latex
\begin{thebibliography}{25}

\bibitem{kingma2014adam}
D.~P. Kingma and J.~Ba, ``Adam: A method for stochastic optimization,'' in \emph{International Conference on Learning Representations (ICLR)}, 2015.

\bibitem{tieleman2012rmsprop}
T.~Tieleman and G.~Hinton, ``Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude,'' \emph{COURSERA: Neural Networks for Machine Learning}, vol.~4, no.~2, pp. 26--31, 2012.

\bibitem{loshchilov2019decoupled}
I.~Loshchilov and F.~Hutter, ``Decoupled weight decay regularization,'' in \emph{International Conference on Learning Representations (ICLR)}, 2019.

\bibitem{yong2020gradient}
H.~Yong, J.~Huang, X.~Hua, and L.~Zhang, ``Gradient centralization: A new optimization technique for deep neural networks,'' in \emph{European Conference on Computer Vision (ECCV)}, 2020, pp. 635--652.

\bibitem{amari1998natural}
S.-I. Amari, ``Natural gradient works efficiently in learning,'' \emph{Neural Computation}, vol.~10, no.~2, pp. 251--276, 1998.

\bibitem{martens2015optimizing}
J.~Martens and R.~Grosse, ``Optimizing neural networks with Kronecker-factored approximate curvature,'' in \emph{International Conference on Machine Learning (ICML)}, 2015, pp. 2408--2417.

\bibitem{grosse2016kronecker}
R.~Grosse and J.~Martens, ``A Kronecker-factored approximate Fisher matrix for convolution layers,'' in \emph{International Conference on Machine Learning (ICML)}, 2016, pp. 573--582.

\bibitem{osawa2019large}
K.~Osawa, Y.~Tsuji, Y.~Ueno, A.~Naruse, R.~Yokota, and S.~Matsuoka, ``Large-scale distributed second-order optimization using Kronecker-factored approximate curvature for deep convolutional neural networks,'' in \emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2019, pp. 8645--8653.

\bibitem{gupta2018shampoo}
V.~Gupta, T.~Koren, and Y.~Singer, ``Shampoo: Preconditioned stochastic tensor optimization,'' in \emph{International Conference on Machine Learning (ICML)}, 2018, pp. 1842--1850.

\bibitem{yao2021adahessian}
Z.~Yao, A.~Gholami, S.~Shen, M.~Mustafa, K.~Keutzer, and M.~W. Mahoney, ``ADAHESSIAN: An adaptive second order optimizer for machine learning,'' in \emph{AAAI Conference on Artificial Intelligence}, 2021, pp. 10665--10673.

\bibitem{vyas2024soap}
N.~Vyas, D.~Morwani, R.~Zhao, I.~Shapira, D.~Brandfonbrener, L.~Janson, and S.~Kakade, ``SOAP: Improving and stabilizing Shampoo using rank-reduced Fisher inverses,'' in \emph{International Conference on Machine Learning (ICML)}, 2024.

\bibitem{jordan2024muon}
K.~Jordan, Y.~Jin, B.~Barak, et~al., ``Muon: An optimizer for hidden layers in neural networks,'' \emph{Technical Report}, 2024.

\bibitem{agarwal2020disentangling}
N.~Agarwal, R.~Anil, E.~Hazan, T.~Koren, and C.~Zhang, ``Disentangling adaptive gradient methods from learning rate schedules,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2020.

\bibitem{spall1992multivariate}
J.~C. Spall, ``Multivariate stochastic approximation using a simultaneous perturbation gradient approximation,'' \emph{IEEE Transactions on Automatic Control}, vol.~37, no.~3, pp. 332--341, 1992.

\bibitem{nesterov2017random}
Y.~Nesterov and V.~Spokoiny, ``Random gradient-free minimization of convex functions,'' \emph{Foundations of Computational Mathematics}, vol.~17, no.~2, pp. 527--566, 2017.

\bibitem{malladi2023mezo}
S.~Malladi, T.~Gao, E.~Nichani, A.~Damian, J.~D. Lee, D.~Chen, and S.~Arora, ``Fine-tuning language models with just forward passes,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2023.

\bibitem{foret2020sharpness}
P.~Foret, A.~Kleiner, H.~Mobahi, and B.~Neyshabur, ``Sharpness-aware minimization for efficiently improving generalization,'' in \emph{International Conference on Learning Representations (ICLR)}, 2021.

\bibitem{ge2015escaping}
R.~Ge, F.~Huang, C.~Jin, and Y.~Yuan, ``Escaping from saddle points---online stochastic gradient for tensor decomposition,'' in \emph{Conference on Learning Theory (COLT)}, 2015, pp. 797--842.

\bibitem{jin2017escape}
C.~Jin, R.~Ge, P.~Netrapalli, S.~M. Kakade, and M.~I. Jordan, ``How to escape saddle points efficiently,'' in \emph{International Conference on Machine Learning (ICML)}, 2017, pp. 1724--1732.

\bibitem{marquardt1963algorithm}
D.~W. Marquardt, ``An algorithm for least-squares estimation of nonlinear parameters,'' \emph{Journal of the Society for Industrial and Applied Mathematics}, vol.~11, no.~2, pp. 431--441, 1963.

\bibitem{he2016deep}
K.~He, X.~Zhang, S.~Ren, and J.~Sun, ``Deep residual learning for image recognition,'' in \emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 2016, pp. 770--778.

\bibitem{zagoruyko2016wide}
S.~Zagoruyko and N.~Komodakis, ``Wide residual networks,'' in \emph{British Machine Vision Conference (BMVC)}, 2016.

\bibitem{dosovitskiy2020image}
A.~Dosovitskiy, L.~Beyer, A.~Kolesnikov, D.~Weissenborn, X.~Zhai, T.~Unterthiner, M.~Dehghani, M.~Minderer, G.~Heigold, S.~Gelly, J.~Uszkoreit, and N.~Houlsby, ``An image is worth 16x16 words: Transformers for image recognition at scale,'' in \emph{International Conference on Learning Representations (ICLR)}, 2021.

\bibitem{vaswani2017attention}
A.~Vaswani, N.~Shazeer, N.~Parmar, J.~Uszkoreit, L.~Jones, A.~N. Gomez, L.~Kaiser, and I.~Polosukhin, ``Attention is all you need,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2017.

\bibitem{krizhevsky2009learning}
A.~Krizhevsky and G.~Hinton, ``Learning multiple layers of features from tiny images,'' \emph{Technical Report, University of Toronto}, 2009.

\bibitem{cui2019class}
Y.~Cui, M.~Jia, T.-Y. Lin, Y.~Song, and S.~Belongie, ``Class-balanced loss based on effective number of samples,'' in \emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2019, pp. 9268--9277.

\bibitem{cao2019learning}
K.~Cao, C.~Wei, A.~Gaidon, N.~Arechiga, and T.~Ma, ``Learning imbalanced datasets with label-distribution-aware margin loss,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2019.

\bibitem{kunstner2024heavy}
F.~Kunstner, R.~Yadav, A.~Milligan, M.~Schmidt, and A.~Bietti, ``Heavy-tailed class imbalance and why Adam outperforms gradient descent on language models,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2024.

\bibitem{loshchilov2016sgdr}
I.~Loshchilov and F.~Hutter, ``SGDR: Stochastic gradient descent with warm restarts,'' in \emph{International Conference on Learning Representations (ICLR)}, 2017.

\bibitem{akiba2019optuna}
T.~Akiba, S.~Sano, T.~Yanase, T.~Ohta, and M.~Koyama, ``Optuna: A next-generation hyperparameter optimization framework,'' in \emph{ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining (KDD)}, 2019, pp. 2623--2631.

\bibitem{paszke2019pytorch}
A.~Paszke, S.~Gross, F.~Massa, A.~Lerer, J.~Bradbury, G.~Chanan, T.~Killeen, Z.~Lin, N.~Gimelshein, L.~Antiga, et~al., ``PyTorch: An imperative style, high-performance deep learning library,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2019.

\bibitem{zhang2019lookahead}
M.~Zhang, J.~Lucas, J.~Ba, and G.~E. Hinton, ``Lookahead optimizer: k steps forward, 1 step back,'' in \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2019.

\end{thebibliography}
```

---

## 4. BibTeX File Format (if using a `.bib` file)

```bibtex
@article{tieleman2012rmsprop,
  title={Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude},
  author={Tieleman, Tijmen and Hinton, Geoffrey},
  journal={COURSERA: Neural networks for machine learning},
  volume={4},
  number={2},
  pages={26--31},
  year={2012}
}

@inproceedings{grosse2016kronecker,
  title={A Kronecker-factored approximate Fisher matrix for convolution layers},
  author={Grosse, Roger and Martens, James},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={573--582},
  year={2016}
}

@inproceedings{gupta2018shampoo,
  title={Shampoo: Preconditioned stochastic tensor optimization},
  author={Gupta, Vineet and Koren, Tomer and Singer, Yoram},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={1842--1850},
  year={2018}
}

@inproceedings{yao2021adahessian,
  title={ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning},
  author={Yao, Zhewei and Gholami, Amir and Shen, Sheng and Mustafa, Mustafa and Keutzer, Kurt and Mahoney, Michael W},
  booktitle={AAAI Conference on Artificial Intelligence},
  pages={10665--10673},
  year={2021}
}

@inproceedings{vyas2024soap,
  title={SOAP: Improving and Stabilizing Shampoo using Rank-Reduced Fisher Inverses},
  author={Vyas, Nikhil and Morwani, Depen and Zhao, Rosie and Shapira, Itai and Brandfonbrener, David and Janson, Lucas and Kakade, Sham},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

@article{jordan2024muon,
  title={Muon: An optimizer for hidden layers in neural networks},
  author={Jordan, Keller and Jin, Yuchen and Barak, Boaz and others},
  journal={Technical Report},
  year={2024}
}

@inproceedings{agarwal2020disentangling,
  title={Disentangling Adaptive Gradient Methods from Learning Rate Schedules},
  author={Agarwal, Naman and Anil, Rohan and Hazan, Elad and Koren, Tomer and Zhang, Cyril},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@inproceedings{malladi2023mezo,
  title={Fine-Tuning Language Models with Just Forward Passes},
  author={Malladi, Sadhika and Gao, Tianyu and Nichani, Eshaan and Damian, Alex and Lee, Jason D and Chen, Danqi and Arora, Sanjeev},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}

@inproceedings{foret2020sharpness,
  title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{ge2015escaping,
  title={Escaping from Saddle Points---Online Stochastic Gradient for Tensor Decomposition},
  author={Ge, Rong and Huang, Furong and Jin, Chi and Yuan, Yang},
  booktitle={Conference on Learning Theory (COLT)},
  pages={797--842},
  year={2015}
}

@inproceedings{jin2017escape,
  title={How to Escape Saddle Points Efficiently},
  author={Jin, Chi and Ge, Rong and Netrapalli, Praneeth and Kakade, Sham M and Jordan, Michael I},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={1724--1732},
  year={2017}
}

@article{marquardt1963algorithm,
  title={An algorithm for least-squares estimation of nonlinear parameters},
  author={Marquardt, Donald W},
  journal={Journal of the Society for Industrial and Applied Mathematics},
  volume={11},
  number={2},
  pages={431--441},
  year={1963}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016}
}

@inproceedings{zagoruyko2016wide,
  title={Wide Residual Networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2016}
}

@inproceedings{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}

@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  journal={Technical Report, University of Toronto},
  year={2009}
}

@inproceedings{cui2019class,
  title={Class-Balanced Loss Based on Effective Number of Samples},
  author={Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9268--9277},
  year={2019}
}

@inproceedings{cao2019learning,
  title={Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss},
  author={Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}

@inproceedings{loshchilov2016sgdr,
  title={SGDR: Stochastic Gradient Descent with Warm Restarts},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@inproceedings{akiba2019optuna,
  title={Optuna: A Next-generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining (KDD)},
  pages={2623--2631},
  year={2019}
}

@inproceedings{paszke2019pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

---

## 5. Audit of Existing Bibliography Entries in `conference_101719.tex`

1. `\bibitem{kunstner2024heavy}`: Currently defined in the reference section, but **never cited** in the paper. 
   * *Recommendation*: Insert into Section IV-A (Line 950) where heavy long-tailed label imbalances are introduced.
2. `\bibitem{zhang2019lookahead}`: Currently defined in the reference section, but **never cited**.
   * *Recommendation*: Insert into Section II-B (Line 106) discussing multi-step / outer-loop stabilization heuristics alongside grafting.
3. `\bibitem{sst2}`: Currently defined in the reference section, but **never cited**. 
   * *Recommendation*: Remove if the paper solely benchmarks computer vision datasets (CIFAR-10 LT).
4. Commented-out `\bibitem{foret2020sharpness}` and `\bibitem{kwon2021asam}`: 
   * *Recommendation*: Uncomment and link `foret2020sharpness` directly to the SAM discussion in Section II-C (Line 109).
