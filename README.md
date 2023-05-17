# Blind_MaxEnt
##  Installation Requirements
We have provided a environment.yml in our repository. 
Main libraries needed: torch, torchvision, numpy, sklearn and a CUDA supported machine.
## Usage
bash run.sh #This will start a for-loop across three random seeds for all nine methods described in the main paper.

## Current support ##
We provide the training scripts for CIFAR.
To evaluate on CIFARC, you have to download it from: https://github.com/hendrycks/robustness
As TinyImageNet-C and Wilds benchmarks are larger, additional support scripts for TinyImageNet-C and Wilds will be included soon.

## Code filepath explanation 
>1.)The main script of interest is in "train_calibration_maxent.py".
>2.)MaxEnt Loss and the other loss functions are implemented in "losses.py"
>3.)The Lagrange multipliers are computed with our implementation of the Newton Raphson method in "maxent_newton_solver.py" which requires only CPU.
