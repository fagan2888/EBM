#!/bin/bash

sim_dir=~/my-scratch/EBM_sims/
# sim_dir=~/ResearchBoos/EBM_files/EBM_sims/

N_pts=401
dtmax_multiple=200
# dtmax_multiple=1e3
# max_sim_years=5
max_sim_years=10
# tol=1e-8
tol=1e-9

initial_condition=legendre
low=250
high=300

albedo_feedback=True
# albedo_feedback=False
alb_ice=0.6
alb_water=0.2

numerical_method=implicit
frames=100

fname_efe=itcz.log
fname_feedbacks=feedbacks.log

if [ "$1" == "-o" ]; then
	# ONE SIMULATION 

	# insolation_type=annual_mean_clark
	insolation_type=perturbation
	# perturb_center=None
	perturb_center=15
	# perturb_center=60
	# perturb_spread=None
	perturb_spread=4.94
	# perturb_spread=9.89
	# perturb_intensity=None
	# perturb_intensity=5
	perturb_intensity=0
	# perturb_intensity=15
	# perturb_intensity=18
	
	olr_type=full_wvf
	# olr_type=full_no_wvf
	A=None
	B=None
	emissivity=None
	# RH_vert_profile=zero_top
	RH_vert_profile=steps
	# RH_lat_profile=constant
	# RH_lat_profile=gaussian
	RH_lat_profile=mid_level_gaussian
	# RH_lat_profile=mid_and_upper_level_gaussian
	gaussian_spread=5
	
	i=0
	while [ -d ${sim_dir}sim$i ];
	do
	    i=`echo "$i + 1" | bc`
	done
	
	echo "Making new simulation in ${sim_dir}sim$i"
	mkdir ${sim_dir}sim$i
	cd ${sim_dir}sim$i
	
	# echo "Copying template files."
	sed -e 's/N_pts=/N_pts='$N_pts'/g' \
	    -e 's/dtmax_multiple=/dtmax_multiple='$dtmax_multiple'/g' \
	    -e 's/max_sim_years=/max_sim_years='$max_sim_years'/g' \
	    -e 's/tol=/tol='$tol'/g' \
	    -e 's/initial_condition=/initial_condition="'$initial_condition'"/g' \
	    -e 's/low=/low='$low'/g' \
	    -e 's/high=/high='$high'/g' \
	    -e 's/albedo_feedback=/albedo_feedback='$albedo_feedback'/g' \
	    -e 's/alb_ice=/alb_ice='$alb_ice'/g' \
	    -e 's/alb_water=/alb_water='$alb_water'/g' \
	    -e 's/insolation_type=/insolation_type="'$insolation_type'"/g' \
	    -e 's/perturb_center=/perturb_center='$perturb_center'/g' \
	    -e 's/perturb_spread=/perturb_spread='$perturb_spread'/g' \
	    -e 's/perturb_intensity=/perturb_intensity='$perturb_intensity'/g' \
	    -e 's/olr_type=/olr_type="'$olr_type'"/g' \
	    -e 's/A=/A='$A'/g' \
	    -e 's/B=/B='$B'/g' \
	    -e 's/emissivity=/emissivity='$emissivity'/g' \
	    -e 's/RH_vert_profile=/RH_vert_profile="'$RH_vert_profile'"/g' \
	    -e 's/RH_lat_profile=/RH_lat_profile="'$RH_lat_profile'"/g' \
	    -e 's/gaussian_spread=/gaussian_spread='$gaussian_spread'/g' \
	    -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
	    -e 's/frames=/frames='$frames'/g' \
	    -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
	    -e 's/fname_feedbacks=/fname_feedbacks="'$fname_feedbacks'"/g' \
	    ${EBM_PATH}/simulation.py > simulation.py
	
	# sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job
	cp -p ${EBM_PATH}/EBM.py .
	
	# echo "Running job."
	# sbatch run_EBM.job
	python -u simulation.py > out0 &
	
	# echo "Logging simulation."
	log="sim$i | lat0=$perturb_center | M=$perturb_intensity | $olr_type |" 
	echo "Adding line to log.txt: $log"
	echo $log >> ${EBM_PATH}/log.txt 
	
	cd ..
elif [ "$1" == "-s" ]; then
	# SENSITIVITY EXPERIMENTS 
	
	olr_type=full_radiation
	# olr_type=full_radiation_no_wv
	# olr_type=full_radiation_no_lr
	# olr_type=planck
	# olr_type=linear
	# A=-572.3
	# B=2.92
	A=None
	B=None
	emissivity=None
	
	insolation_type=perturbation

	i=282
	# i=0
	# while [ -d ${sim_dir}sim$i ];
	# do
	#     i=`echo "$i + 1" | bc`
	# done
	# echo "Making simulations in ${sim_dir}sim$i"

	# mkdir ${sim_dir}sim$i
	mkdir ${sim_dir}sim${i}/tropical
	mkdir ${sim_dir}sim${i}/extratropical

	for perturb_center in 15 60; do
        if [ $perturb_center -eq 15 ]; then
            perturb_spread=4.94
        else
            perturb_spread=9.89
        fi
	    for perturb_intensity in 5 10 15 18; do
        	if [ $perturb_center -eq 15 ]; then
				dir=${sim_dir}sim${i}/tropical/M$perturb_intensity
        	else
				dir=${sim_dir}sim${i}/extratropical/M$perturb_intensity
        	fi
	        echo "Making new simulation in $dir"
	        mkdir $dir
	        cd $dir
	        
	        # echo "Copying template files."
	        sed -e 's/N_pts=/N_pts='$N_pts'/g' \
	            -e 's/dtmax_multiple=/dtmax_multiple='$dtmax_multiple'/g' \
	            -e 's/max_sim_years=/max_sim_years='$max_sim_years'/g' \
	            -e 's/tol=/tol='$tol'/g' \
	            -e 's/initial_condition=/initial_condition="'$initial_condition'"/g' \
	            -e 's/low=/low='$low'/g' \
	            -e 's/high=/high='$high'/g' \
	            -e 's/albedo_feedback=/albedo_feedback='$albedo_feedback'/g' \
	            -e 's/alb_ice=/alb_ice='$alb_ice'/g' \
	            -e 's/alb_water=/alb_water='$alb_water'/g' \
	            -e 's/insolation_type=/insolation_type="'$insolation_type'"/g' \
	            -e 's/perturb_center=/perturb_center='$perturb_center'/g' \
	            -e 's/perturb_spread=/perturb_spread='$perturb_spread'/g' \
	            -e 's/perturb_intensity=/perturb_intensity='$perturb_intensity'/g' \
	            -e 's/olr_type=/olr_type="'$olr_type'"/g' \
	            -e 's/A=/A='$A'/g' \
	            -e 's/B=/B='$B'/g' \
	            -e 's/emissivity=/emissivity='$emissivity'/g' \
	            -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
	            -e 's/frames=/frames='$frames'/g' \
	            -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
	            -e 's/fname_feedbacks=/fname_feedbacks="'$fname_feedbacks'"/g' \
	            ${EBM_PATH}/simulation.py > simulation.py
	        
	        sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job
	        cp -p ${EBM_PATH}/EBM.py .
	        
	        # echo "Running job."
	        sbatch run_EBM.job
			# python -u simulation.py > out0 &
	        
	        # echo "Logging simulation."
	        log="sim$i | $insolation_type | lat0=$perturb_center | M=$perturb_intensity | $olr_type | "
	        echo "Adding line to log.txt: $log"
	        echo $log >> ${EBM_PATH}/log.txt 
	        
	        cd ..
        done
    done
elif [ "$1" == "-c" ]; then
	# CONVERGENCE TESTS 
	insolation_type=perturbation
	# perturb_center=None
	perturb_center=15
	# perturb_center=60
	# perturb_spread=None
	perturb_spread=4.94
	# perturb_spread=9.89
	# perturb_intensity=None
	perturb_intensity=0
	# perturb_intensity=5
	# perturb_intensity=10
	# perturb_intensity=15
	# perturb_intensity=18
	
	# olr_type=full_wvf
	# olr_type=full_no_wvf
	olr_type=planck
	A=None
	B=None
	# emissivity=None
	emissivity=0.6
	# RH_vert_profile=zero_top
	RH_vert_profile=steps
	# RH_lat_profile=constant
	# RH_lat_profile=gaussian
	RH_lat_profile=mid_level_gaussian
	# RH_lat_profile=mid_and_upper_level_gaussian
	gaussian_spread=5
	
	i=0
	while [ -d ${sim_dir}sim$i ];
	do
	    i=`echo "$i + 1" | bc`
	done
	
	for N_pts in 401 501; do
		for tol in 1e-8 1e-9 1e-10; do
			for dtmax_multiple in 0.1 1.0 5.0 10.0 50.0; do
				echo "Making new simulation in ${sim_dir}sim$i"
				mkdir ${sim_dir}sim$i
				cd ${sim_dir}sim$i
				
				# echo "Copying template files."
				sed -e 's/N_pts=/N_pts='$N_pts'/g' \
				    -e 's/dtmax_multiple=/dtmax_multiple='$dtmax_multiple'/g' \
				    -e 's/max_sim_years=/max_sim_years='$max_sim_years'/g' \
				    -e 's/tol=/tol='$tol'/g' \
				    -e 's/initial_condition=/initial_condition="'$initial_condition'"/g' \
				    -e 's/low=/low='$low'/g' \
				    -e 's/high=/high='$high'/g' \
				    -e 's/albedo_feedback=/albedo_feedback='$albedo_feedback'/g' \
				    -e 's/alb_ice=/alb_ice='$alb_ice'/g' \
				    -e 's/alb_water=/alb_water='$alb_water'/g' \
				    -e 's/insolation_type=/insolation_type="'$insolation_type'"/g' \
				    -e 's/perturb_center=/perturb_center='$perturb_center'/g' \
				    -e 's/perturb_spread=/perturb_spread='$perturb_spread'/g' \
				    -e 's/perturb_intensity=/perturb_intensity='$perturb_intensity'/g' \
				    -e 's/olr_type=/olr_type="'$olr_type'"/g' \
				    -e 's/A=/A='$A'/g' \
				    -e 's/B=/B='$B'/g' \
				    -e 's/emissivity=/emissivity='$emissivity'/g' \
				    -e 's/RH_vert_profile=/RH_vert_profile="'$RH_vert_profile'"/g' \
				    -e 's/RH_lat_profile=/RH_lat_profile="'$RH_lat_profile'"/g' \
				    -e 's/gaussian_spread=/gaussian_spread='$gaussian_spread'/g' \
				    -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
				    -e 's/frames=/frames='$frames'/g' \
				    -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
				    -e 's/model.log_feedbacks/# model.log_feedbacks/g' \
				    ${EBM_PATH}/simulation.py > simulation.py
				
				chmod 755 simulation.py
				# sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job

				cp -p ${EBM_PATH}/EBM.py .
				
				# echo "Running job."
				# sbatch run_EBM.job
				python -u simulation.py > out0 &
				
				# echo "Logging simulation."
				log="sim$i | N_pts=$N_pts | dtmax_multiple=$dtmax_multiple | tol=$tol | "
				echo "Adding line to log.txt: $log"
				echo $log >> ${EBM_PATH}/log.txt 
				
				cd ..
	    		i=`echo "$i + 1" | bc`
			done
		done
	done
else
	echo "Please give an argument."
fi
