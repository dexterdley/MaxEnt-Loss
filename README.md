# Blind_MaxEnt
##  Installation Requirements
We have provided a environment.yml in our repository. However, there may be some unnecessary libraries included. 
Main libraries needed: torch, torchvision, numpy, sklearn and a CUDA supported machine.
## Usage
bash run.sh

## Code filepath explanation 
>1.)The main script of interest is in "train_calibration_maxent_grid_search.py", where MaxEnt loss is implemented.

>2.)The Lagrange multipliers are computed with our implementation of the Newton Raphson method in "maxent_newton_solver.py" using only CPU.
