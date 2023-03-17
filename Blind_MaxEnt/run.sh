#cifar10
python train_calibration_maxent_grid_search.py --expt_name=cifar10_calibration_onehot --constraints=0 --gamma=0 --arch='res18' --shift_ood='T1' --seed=1
#python train_calibration_maxent_grid_search.py --expt_name=cifar10_calibration_shannon --constraints=0 --gamma=1 --arch='res18' --shift_ood='T1' --seed=1
#python train_calibration_maxent_grid_search.py --expt_name=cifar10_calibration_poly --constraints=3 --gamma=0 --arch='res18' --shift_ood='T1' --seed=1
#python train_calibration_maxent_grid_search.py --expt_name=cifar10_calibration_max_ent_mu --constraints=1 --gamma=1 --arch='res18' --shift_ood='T1' --seed=1
#python train_calibration_maxent_grid_search.py --expt_name=cifar10_calibration_max_ent_var --constraints=2 --gamma=1 --arch='res18' --shift_ood='T1' --seed=1
#python train_calibration_maxent_grid_search.py --expt_name=cifar100_calibration_IFL --constraints=5 --gamma=1 --arch='res18' --shift_ood='T1' --seed=1

