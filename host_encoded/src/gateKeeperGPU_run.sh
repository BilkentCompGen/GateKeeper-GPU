#!/bin/bash

# Modify only these:
# file_input="../../../../../inputs/12Datasets_Zulal/ERR240727_1_E2_30million.txt"
file_input="../../../../inputs/SRR826460_1_E10_30million.txt"
file_name="SRR826460_1_E10_30million"
read_length=150
error_threshold=0
maxError=6
maxGPU=4
clean_outputs=true #change this if you want output files after the test

#DO NOT CHANGE ANYTHING AFTER THIS LINE:
sed -i "s/^READ_LENGTH = .*/READ_LENGTH = ${read_length}/" makefile

# while [ $error_threshold -le $maxError ]
for error_threshold in 0 1 2 3 4 5 6 7 8 9 10
do
	sed -i "s/^ERROR_THRESHOLD = .*/ERROR_THRESHOLD = ${error_threshold}/" makefile
	make
	testID="${file_name}_${read_length}bp_err${error_threshold}"

	{

	gpuCount=1
	echo --------------------------------------------------------------------------------GPU TEST FOR E = $error_threshold

	while [ $gpuCount -le $maxGPU ]
	do
		echo RUNNER OPTIONS:
		echo GPU = $gpuCount
		echo debug = $1
		echo execount = $2
		echo Test ID: GPU_$testID

		rm -rf infile.log
		counter=1

		while [ $counter -le $2 ]
		do
			echo ----Test = num device: $gpuCount, test: $counter
			./gateKeeperGPU $file_input $gpuCount $1 | tee -a infile.log    
			((counter++))
		done

		echo "**********"
		awk -v execount=$2 -F':' '/^Kernel /{s+=$2} END {print "kernel sum:", s; avg=s/execount; print "kernel avg:", avg}' infile.log
		awk -v execount=$2 -F':' '/^Only /{s+=$2} END {print "Exe sum:", s; avg=s/execount; print "Exe avg:", avg}' infile.log
		awk -v execount=$2 -F':' '/^Overall /{s+=$2} END {print "end-to-end sum:", s; avg=s/execount; print "end-to-end avg:", avg}' infile.log
		echo "**********"

		# Clean for testing :
		if [ "$clean_outputs" = true ]; then
			file_output_filter="gateKeeperGPU_${gpuCount}G_filter_output_err${error_threshold}.txt"
			file_output_edit="gateKeeperGPU_${gpuCount}G_edit_output_err${error_threshold}.txt"		
			rm -rf $file_output_edit
			rm -rf $file_output_filter
		fi

	((gpuCount++))
	done

	} > $testID.log

	echo PROGRAM COMPLETED FOR E = $error_threshold ---------------------------------------------------------------------------

	# ((error_threshold++))
	rm -rf infile.log
	make clean

done
echo ALL DONE ---------------------------------------------------------------------------



