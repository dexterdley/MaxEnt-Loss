# MaxEnt Loss: Constrained Maximum Entropy for Calibration under Out-of-Distribution Shift (AAAI 24) [Oral Presentation]
Official code implementation for the paper MaxEnt Loss: Constrained Maximum Entropy for Calibration under Out-of-Distribution Shift

Authors: Dexter Neo, Stefan Winkler, Tsuhan Chen \
URL: [https://ojs.aaai.org/index.php/AAAI/article/view/30143](https://ojs.aaai.org/index.php/AAAI/article/view/30143) \
Arxiv: [https://arxiv.org/abs/2310.17159](https://arxiv.org/abs/2310.17159)

##  Installation Requirements
We have provided a environment.yml in our repository. 
Main libraries needed: torch, torchvision, numpy, sklearn and a CUDA supported machine.
## Usage
Call bash run.sh #This will start a for-loop across three random seeds for all nine methods described in the main paper.

## Current support ##
We provide the training scripts for CIFAR.
To evaluate on CIFARC, you have to download it from: https://github.com/hendrycks/robustness
As TinyImageNet-C and Wilds benchmarks are larger, additional support scripts for TinyImageNet-C and Wilds will be included soon.

## Code filepath explanation 
>1.)The main script of interest is in "train_calibration_maxent.py". \
>2.)MaxEnt Loss and the other loss functions are implemented in "losses.py"\
>3.)The Lagrange multipliers are computed with our implementation of the Newton Raphson method in "maxent_newton_solver.py" which requires only CPU.

## OOD Model Calibration with MaxEnt Loss
![Curves](https://github.com/dexterdley/Blind_MaxEnt/blob/main/figures/cifarc_plots_2.png)

Under the influence of OOD, many objective functions become miscalibrated. In this work we propose considering constraints observed from the training set to improve OOD calibration. We show that MaxEnt loss delivers well calibrated models OOD, for more details please refer to the main paper.

![Reliability](https://github.com/dexterdley/Blind_MaxEnt/blob/main/figures/cifarc_bin_strength_reliability.png)
