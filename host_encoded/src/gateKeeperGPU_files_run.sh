#!/bin/bash

# Modify only these:

files="../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E0_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E1_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E2_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E3_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E4_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E5_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E6_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E7_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E8_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E9_30million.seq
../../../../../inputs/minimap2_samples/minimap2_ERR240727_1_E10_30million.seq"

file_name="minimap2_ERR240727_1"
read_length=100
error_threshold=0
maxError=10
maxGPU=1
clean_outputs=false #change this if you want output files after the test

#DO NOT CHANGE ANYTHING AFTER THIS LINE:
sed -i "s/^READ_LENGTH = .*/READ_LENGTH = ${read_length}/" makefile

for file in $files
do
	sed -i "s/^ERROR_THRESHOLD = .*/ERROR_THRESHOLD = ${error_threshold}/" makefile
	make
	testID="GK-GPUtesla_${file_name}_${read_length}bp_err${error_threshold}"
	echo "Processing $file"

	{

	gpuCount=1
	echo --------------------------------------------------------------------------------GPU TEST FOR E = $error_threshold

	while [ $gpuCount -le $maxGPU ]
	do
		echo RUNNER OPTIONS:
		echo GPU = $gpuCount
		echo debug = $1
		echo execount = $2
		echo Test ID: $testID

		rm -rf infile.log
		counter=1

		while [ $counter -le $2 ]
		do
			echo ----Test = GPU: $gpuCount, test: $counter
			./gateKeeperGPU $file $gpuCount $1 | tee -a infile.log    
			((counter++))
		done

		echo "**********"
		awk -v execount=$2 -F':     ' '/Overall /{s+=$2} END {print "end-to-end sum:", s; avg=s/execount; print "end-to-end avg:", avg}' infile.log
		awk -v execount=$2 -F':     ' '/Only /{s+=$2} END {print "Exe sum:", s; avg=s/execount; print "Exe avg:", avg}' infile.log
		echo "**********"

		# Clean for testing :
		if [ "$clean_outputs" = true ]; then
			file_output_filter="gateKeeperGPU_${threadCount}G_filter_output_err${error_threshold}.txt"
			file_output_edit="gateKeeperGPU_${threadCount}G_edit_output_err${error_threshold}.txt"		
			rm -rf $file_output_edit
			rm -rf $file_output_filter
		fi

	((gpuCount++))
	done

	} > $testID.log

	echo PROGRAM COMPLETED FOR E = $error_threshold ---------------------------------------------------------------------------

	((error_threshold++))
	rm -rf infile.log
	make clean

done
echo ALL DONE ---------------------------------------------------------------------------



