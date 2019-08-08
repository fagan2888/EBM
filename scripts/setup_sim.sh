#!/bin/bash

# sim_dir=~/my-scratch/EBM_sims/
sim_dir=~/ResearchBoos/EBM_files/EBM_sims/

N_pts=513
max_iters=1e3
tol=1e-4
# diffusivity=constant
# diffusivity=cesm2
# diffusivity=D1
diffusivity=D2
control_file=default

init_temp_type=legendre
low=250
high=300

al_feedback=True
# al_feedback=False
alb_ice=0.6
alb_water=0.2

numerical_method=multigrid
frames=100

control=False

fname_efe=itcz.log
fname_feedbacks=feedbacks.log

if [ "$1" == "-s" ]; then
	# SENSITIVITY EXPERIMENTS 
	olr_type=full_radiation
	# olr_type=full_radiation_no_wv
	# olr_type=full_radiation_no_lr
	# olr_type=full_radiation_rh
	# olr_type=no_feedback
	# olr_type=full_radiation_no_wv_no_lr
	A=None
	B=None
	emissivity=None
	
	insol_type=perturbation

    i=291
	# i=0
	# while [ -d ${sim_dir}sim$i ];
	# do
	#     i=`echo "$i + 1" | bc`
	# done
	echo "Making simulations in ${sim_dir}sim$i"

	mkdir ${sim_dir}sim$i
	if [ $diffusivity != constant ]; then
		sim_name=$diffusivity
	else 
		if [ $al_feedback = False ]; then
			sim_name=${olr_type}_no_al
		else
			sim_name=${olr_type}
		fi
	fi
	mkdir ${sim_dir}sim${i}/$sim_name

	for perturb_center in 15 60; do
        if [ $perturb_center -eq 15 ]; then
            perturb_spread=4.94
        else
            perturb_spread=9.89
        fi
	    for perturb_intensity in 5 10 15 18; do
        	if [ $perturb_center -eq 15 ]; then
                subdir=T
        	else
                subdir=E
        	fi
            if [ $perturb_intensity -eq 5 ]; then
			    dir=${sim_dir}sim${i}/${sim_name}/${subdir}0$perturb_intensity
            else
			    dir=${sim_dir}sim${i}/${sim_name}/${subdir}$perturb_intensity
            fi
	        mkdir $dir
	        cd $dir
	        
	        # echo "Copying template files."
			echo "import mebm" >> simulation.py
			echo 'model = mebm.MoistEnergyBalanceModel(N_pts='$N_pts', max_iters='$max_iters', tol='$tol', diffusivity="'$diffusivity'", control_file="'$control_file'")' >> simulation.py
			echo 'model.set_init_temp(init_temp_type="'$init_temp_type'", low='$low', high='$high')' >> simulation.py
			echo 'model.set_insol(insol_type="'$insol_type'", perturb_center='$perturb_center', perturb_spread='$perturb_spread', perturb_intensity='$perturb_intensity')' >> simulation.py
			echo 'model.set_albedo(al_feedback='$al_feedback', alb_ice='$alb_ice', alb_water='$alb_water')' >> simulation.py
			echo 'model.set_olr(olr_type="'$olr_type'", A='$A', B='$B', emissivity='$emissivity')' >> simulation.py
			echo 'model.solve(numerical_method="'$numerical_method'", frames='$frames')' >> simulation.py
			echo 'model.save_data(control='$control')' >> simulation.py
			echo 'model.log_efe(fname_efe="'$fname_efe'")' >> simulation.py
			echo 'model.log_feedbacks(fname_feedbacks="'$fname_feedbacks'")' >> simulation.py
			echo 'model.save_plots()' >> simulation.py
	        
	        # echo "Running job."
	        # sbatch run_EBM.job
			nohup python -u simulation.py > out0 2> out.err &
	        
	        # echo "Logging simulation."
	        echo "sim$i | lat0=$perturb_center | M=$perturb_intensity | $olr_type | al_feedback=$al_feedback | $diffusivity"

	        cd ..
        done
    done
else
	echo "Please give an argument."
fi
