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
# diffusivity=constant
# diffusivity=cesm2
diffusivity=D1
# diffusivity=D2

initial_condition=legendre
low=250
high=300

albedo_feedback=True
# albedo_feedback=False
alb_ice=0.6
alb_water=0.2

numerical_method=implicit
frames=100

control=False

fname_efe=itcz.log
fname_feedbacks=feedbacks.log

if [ "$1" == "-s" ]; then
	# SENSITIVITY EXPERIMENTS 
	
	# olr_type=full_radiation
	olr_type=full_radiation
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
	            -e 's/diffusivity=/diffusivity="'$diffusivity'"/g' \
	            -e 's/initial_condition=/initial_condition="'$initial_condition'"/g' \
	            -e 's/low=/low='$low'/g' \
	            -e 's/high=/high='$high'/g' \
	            -e 's/al_feedback=/al_feedback='$albedo_feedback'/g' \
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
	            -e 's/control=/control='$control'/g' \
	            -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
	            -e 's/fname_feedbacks=/fname_feedbacks="'$fname_feedbacks'"/g' \
	            ${EBM_PATH}/simulation.py > simulation.py
	        
	        sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job
	        cp -p ${EBM_PATH}/EBM.py .
	        
	        # echo "Running job."
	        sbatch run_EBM.job
			# python -u simulation.py > out0 &
	        
	        # echo "Logging simulation."
	        log="sim$i | $insolation_type | lat0=$perturb_center | M=$perturb_intensity | $olr_type | albedo_feedback=$albedo_feedback"
	        echo "Adding line to log.txt: $log"
	        echo $log >> ${EBM_PATH}/log.txt 
	        
	        cd ..
        done
    done
else
	echo "Please give an argument."
fi
