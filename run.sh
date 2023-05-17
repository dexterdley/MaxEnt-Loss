for i in 0 1 2
do
	#cifar10
	python train_calibration_maxent.py --expt_name=cifar10_CE --constraints=0 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_Focal --constraints=1 --gamma=1 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_InvFocal --constraints=2 --gamma=2 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_AUAvU --constraints=3 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_SoftAUAvU --constraints=4 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_Poly --constraints=5 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_MaxEntMu --constraints=6 --gamma=1 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_MaxEntVar --constraints=7 --gamma=1 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar10_MaxEntMult --constraints=8 --gamma=1 --arch='res18' --seed=$i

	#cifar100
	python train_calibration_maxent.py --expt_name=cifar100_CE --constraints=0 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_Focal --constraints=1 --gamma=1 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_InvFocal --constraints=2 --gamma=2 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_AUAvU --constraints=3 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_SoftAUAvU --constraints=4 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_Poly --constraints=5 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_MaxEntMu --constraints=6 --gamma=1 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_MaxEntVar --constraints=7 --gamma=1 --arch='res18' --seed=$i
	python train_calibration_maxent.py --expt_name=cifar100_MaxEntMult --constraints=8 --gamma=1 --arch='res18' --seed=$i
done