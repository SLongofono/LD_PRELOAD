#
# FILE		: nv_profile.sh
# 
# BRIEF		: This script invokes the matrix-multiply cuda kernel and
# 		  profiles it with selected CLI options offered by nvprof utility
#
# AUTHOR	: Waqar Ali (https://github.com/Skidro)
#

# Declare unicode colors numbers
RED='\033[0;31m'
GRN='\033[0;32m'
NCC='\033[0m'

# Declare the CLI options of interest
declare -a nvprof_events=("--cpu-thread-tracing on"
			  "--cpu-thread-tracing off"
			  "--profile-all-processes"
			  "--profile-api-trace none"
			  "--profile-api-trace runtime"
			  "--profile-api-trace driver"
			  "--profile-api-trace all"
			  "--cpu-profiling on"
			  "--cpu-profiling off"
			  "--cpu-profiling-mode flat"
			  "--cpu-profiling-mode top-down"
			  "--cpu-profiling-mode bottom-up"
			  "--cpu-profiling-scope function"
			  "--cpu-profiling-scope instruction"
			  "--cpu-profiling-show-library on"
			  "--cpu-profiling-show-library off"
			  "--cpu-profiling-thread-mode on"
			  "--cpu-profiling-thread-mode off"
			  "--print-api-summary"
			  "--print-api-trace"
			  "--print-gpu-summary"
			  "--print-gpu-trace"
			  "--print-nvlink-topology"
			  )

# Decalre the name of cuda application
cuda_app=matrixMul

# Specify the name of output directory
output_dir=nvprof

# Specify the name of checker script
check_script=check.py

# Run the cuda kerenl for each CLI option
for option in "${nvprof_events[@]}"; do
	# Create an appropriate log-file name based on the CLI option
	file_name=$(echo $option  | sed "s/\s/-/" | sed "s/--//")

	# Display the progress
	echo -e "${GRN}Profiling $cuda_app with $option${NCC}"

	# Execute the kernel
	nvprof $option --log-file $output_dir/$file_name.log ../$cuda_app &> $output_dir/$file_name.perf

	# Check the output
	python $check_script $output_dir/$file_name.perf

	# Display an empty line
	echo
done
