#!/bin/bash

sim_dir=~/my-scratch/EBM_sims/
# sim_dir=~/Research2018/EBM_files/EBM_sims/

dlat=0.5
dtmax_multiple=1.0
max_iters=1e5
tol=0.001

initial_condition=triangle
triangle_low=270
triangle_high=305

albedo_feedback=False
alb_ice=None
alb_water=None

insolation_type=perturbation
perturb_center=15
# perturb_center=60
perturb_spread=4.94
# perturb_spread=9.89
perturb_intensity=15

olr_type=full_wvf
A=None
B=None
emissivity=None
RH_vert_profile=steps
RH_lat_profile=mid_and_upper_level_gaussian
scale_efe=True

numerical_method=crank
nPlot=100
nPrint=500

fname="efe.log"

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
    -e 's/max_iters=/max_iters='$max_iters'/g' \
    -e 's/tol=/tol='$tol'/g' \
    -e 's/initial_condition=/initial_condition="'$initial_condition'"/g' \
    -e 's/triangle_low=/triangle_low='$triangle_low'/g' \
    -e 's/triangle_high=/triangle_high='$triangle_high'/g' \
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
    -e 's/scale_efe=/scale_efe="'$scale_efe'"/g' \
    -e 's/numerical_method=/numerical_method="'$numerical_method'"/g' \
    -e 's/nPlot=/nPlot='$nPlot'/g' \
    -e 's/nPrint=/nPrint='$nPrint'/g' \
    -e 's/fname=/fname="'$fname'"/g' \
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
# echo "Done."
