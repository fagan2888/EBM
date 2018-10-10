#!/bin/bash

sim_dir=~/my-scratch/EBM_sims/
# sim_dir=~/Research2018/EBM_files/EBM_sims/

dlat=0.5
dtmax_multiple=1.5
max_sim_years=5
tol=1e-8

initial_condition=legendre
low=270
high=305

albedo_feedback=False
alb_ice=None
alb_water=None

numerical_method=implicit
frames=100

fname_efe=itcz.log
fname_feedbacks=feedbacks.log

if [ "$1" == "-p" ]; then
	# SINGLE CONTROL SIMULATION
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
	perturb_intensity=10
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
	# RH_lat_profile=mid_level_gaussian
	RH_lat_profile=mid_and_upper_level_gaussian
	gaussian_spread1=10
	gaussian_spread2=45
	# scale_efe=True
	scale_efe=False
	# constant_spec_hum=True
	constant_spec_hum=False
	
	
	i=0
	while [ -d ${sim_dir}sim$i ];
	do
	    i=`echo "$i + 1" | bc`
	done
	
	echo "Making new simulation in ${sim_dir}sim$i"
	mkdir ${sim_dir}sim$i
	cd ${sim_dir}sim$i
	
	# echo "Copying template files."
	sed -e 's/dlat=/dlat='$dlat'/g' \
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
	    -e 's/gaussian_spread1=/gaussian_spread1='$gaussian_spread1'/g' \
	    -e 's/gaussian_spread2=/gaussian_spread2='$gaussian_spread2'/g' \
	    -e 's/scale_efe=/scale_efe='$scale_efe'/g' \
	    -e 's/constant_spec_hum=/constant_spec_hum='$constant_spec_hum'/g' \
	    -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
	    -e 's/frames=/frames='$frames'/g' \
	    -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
	    -e 's/fname_feedbacks=/fname_feedbacks="'$fname_feedbacks'"/g' \
	    ${EBM_PATH}/simulation.py > simulation.py
	
	sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job
	cp -p ${EBM_PATH}/DEBM.py .
	
	# echo "Running job."
	sbatch run_EBM.job
	
	# echo "Logging simulation."
	log="sim$i | $insolation_type | lat0=$perturb_center | M=$perturb_intensity | $olr_type | RH_vert_profile=$RH_vert_profile | RH_lat_profile=$RH_lat_profile | scale_efe=$scale_efe |"
	echo "Adding line to log.txt: $log"
	echo $log >> ${EBM_PATH}/log.txt 
	
	cd ..
elif [ "$1" == "-s" ]; then
	# SENSITIVITY EXPERIMENTS
	
	# olr_type=full_wvf
	olr_type=full_no_wvf
	# olr_type=linear
	A=None
	# A=-452.68
	B=None
	# B=2.51
	emissivity=None
	# RH_vert_profile=None
	# RH_vert_profile=zero_top
	RH_vert_profile=steps
	# RH_lat_profile=None
	# RH_lat_profile=constant
	# RH_lat_profile=gaussian
	RH_lat_profile=mid_level_gaussian
	# RH_lat_profile=mid_and_upper_level_gaussian
	# gaussian_spread1=None
	gaussian_spread1=5
	gaussian_spread2=None
	# gaussian_spread2=45
	# scale_efe=None
	# scale_efe=True
	scale_efe=False
	# constant_spec_hum=True
	constant_spec_hum=False
	
	insolation_type=perturbation
	for perturb_center in 15 60; do
        if [ $perturb_center -eq 15 ]; then
            perturb_spread=4.94
        else
            perturb_spread=9.89
        fi
	    for perturb_intensity in 5 10 15 18; do
	        i=0
	        while [ -d ${sim_dir}sim$i ];
	        do
	            i=`echo "$i + 1" | bc`
	        done
	        
	        echo "Making new simulation in ${sim_dir}sim$i"
	        mkdir ${sim_dir}sim$i
	        cd ${sim_dir}sim$i
	        
	        # echo "Copying template files."
	        sed -e 's/dlat=/dlat='$dlat'/g' \
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
	            -e 's/gaussian_spread1=/gaussian_spread1='$gaussian_spread1'/g' \
	            -e 's/gaussian_spread2=/gaussian_spread2='$gaussian_spread2'/g' \
	            -e 's/scale_efe=/scale_efe='$scale_efe'/g' \
	    		-e 's/constant_spec_hum=/constant_spec_hum='$constant_spec_hum'/g' \
	            -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
	            -e 's/frames=/frames='$frames'/g' \
	            -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
	            -e 's/fname_feedbacks=/fname_feedbacks="'$fname_feedbacks'"/g' \
	            ${EBM_PATH}/simulation.py > simulation.py
	        
	        sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job
	        cp -p ${EBM_PATH}/DEBM.py .
	        
	        # echo "Running job."
	        sbatch run_EBM.job
	        
	        # echo "Logging simulation."
	        log="sim$i | $insolation_type | lat0=$perturb_center | M=$perturb_intensity | $olr_type | RH_vert_profile=$RH_vert_profile | RH_lat_profile=$RH_lat_profile | gaussian_spread1=$gaussian_spread1 |"
	        # log="sim$i | $insolation_type | lat0=$perturb_center | M=$perturb_intensity | $olr_type | A=$A | B=$B |"
	        echo "Adding line to log.txt: $log"
	        echo $log >> ${EBM_PATH}/log.txt 
	        
	        cd ..
        done
    done
elif [ "$1" == "-amc" ]; then
	# ANNUAL MEAN CLARK CONTROL SIMULATION
	insolation_type=annual_mean_clark
	# insolation_type=perturbation
	perturb_center=None
	# perturb_center=15
	# perturb_center=60
	perturb_spread=None
	# perturb_spread=4.94
	# perturb_spread=9.89
	perturb_intensity=None
	# perturb_intensity=5
	# perturb_intensity=10
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
	gaussian_spread1=5
	gaussian_spread2=45
	# scale_efe=True
	scale_efe=False
    # constant_spec_hum=True
    constant_spec_hum=False
	
	i=0
	while [ -d ${sim_dir}sim$i ];
	do
	    i=`echo "$i + 1" | bc`
	done
	
	echo "Making new simulation in ${sim_dir}sim$i"
	mkdir ${sim_dir}sim$i
	cd ${sim_dir}sim$i
	
	# echo "Copying template files."
	sed -e 's/dlat=/dlat='$dlat'/g' \
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
	    -e 's/gaussian_spread1=/gaussian_spread1='$gaussian_spread1'/g' \
	    -e 's/gaussian_spread2=/gaussian_spread2='$gaussian_spread2'/g' \
	    -e 's/scale_efe=/scale_efe='$scale_efe'/g' \
	    -e 's/constant_spec_hum=/constant_spec_hum='$constant_spec_hum'/g' \
	    -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
	    -e 's/frames=/frames='$frames'/g' \
	    -e 's/fname_efe=/fname_efe="'$fname_efe'"/g' \
	    -e 's/fname_feedbacks=/fname_feedbacks="'$fname_feedbacks'"/g' \
	    ${EBM_PATH}/simulation.py > simulation.py
	
	sed -e 's/NAME/'Sim$i'/g' ${EBM_PATH}/run_EBM.job > run_EBM.job
	cp -p ${EBM_PATH}/DEBM.py .
	
	# echo "Running job."
	sbatch run_EBM.job
	
	# echo "Logging simulation."
	log="sim$i | $insolation_type | $olr_type | RH_vert_profile=$RH_vert_profile | RH_lat_profile=$RH_lat_profile | "
	echo "Adding line to log.txt: $log"
	echo $log >> ${EBM_PATH}/log.txt 
	
	cd ..
else
	echo "Please give an argument."
fi
