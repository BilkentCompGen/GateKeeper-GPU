# GateKeeper-GPU: Fast and Accurate Pre-Alignment Filtering in Short Read Mapping
Calculating the similarities and differences between two sequences is a computationally expensive task since approximate string matching techniques traditionally inherit dynamic programming algorithms with quadratic time and space complexity. We introduce GateKeeper-GPU, a fast and accurate pre-alignment filter that efficiently reduces the need for expensive sequence alignment. GateKeeper-GPU improves the filtering accuracy of GateKeeper (Alser et al.), and it concurrently examines numerous sequence pairs rapidly by exploiting the massive parallelism provided by GPU threads. GateKeeper-GPU accelerates the sequence alignment by up to 2.9x and provides up to 1.4x speedup to the end-to-end execution time of a comprehensive read mapper (mrFAST).

## Usage
In preprocessing steps of GateKeeper-GPU, the sequences are encoded in 2-bit format. We provide GateKeeper-GPU in two versions: host-encoded and device-encoded.  Encoding is carried out in CPU in [host-encoded](host_encoded) version and the bulk of encoded sequences are transferred to GPU. In [device-encoded](device_encoded) version, sequences are copied  to GPU in string format and each GPU thread encodes its own sequence pairs. Either one of these versions can be used depending on the read mapper's workflow. For instance, if the mapper already has an encoding mechanism, using host-encoded version can be more suitable.

### Requirements
GateKeeper-GPU requires OpenMP and CUDA-10 with compute capability 3.5 or above. It works best with compute capability 6.x or above, since data prefetching is involved in some stages. 

### Input/Output Format
For now, GateKeeper-GPU accepts input files in which each line contains a single tab-separated sequence pair for comparison: <br>
`[read] [reference segment]` <br>
At the end of the execution, two output files are generated. *Filter file* contains the result of each filtering operation ('1' for pass, '0' for reject) in each line. *Edit file* contains the edit distances approximated by GateKeeper-GPU.  

### Compilation
Before compiling GateKeeper-GPU, in the makefile please change the values of *READ_LENGTH* of reads (line = 39), and *ERROR_THRESHOLD* for the filtering (line = 40). After specifying these values, simply type inside of the version: <br>
```
make
``` 

### Run
To run GateKeeper-GPU: <br>
```
./gateKeeperGPU <input file> <number of GPUs> <verbose> <n_threads>
```
Input file should be in the format specified above. You can specify the number of GPU devices, GateKeeper-GPU supports up to 8 GPUs. *Verbose* mode can be used for debugging purposes. For debugging, please enter '1' in *verbose* option, otherwise enter '0'. n_threads denotes the number of threads for IO operations. In host_encoded version, encoding is also included in multi-threaded IO operations.  

## Citation
Please visit our full paper arXiv preprint of [GateKeeper-GPU](https://arxiv.org/abs/2103.14978) for more design details and analyses. 
You can cite it as:
>  Z. Bingöl, M. Alser, O. Mutlu, O. Ozturk, and C. Alkan, “Gatekeeper-GPU: Fast and Accurate Pre-alignment Filtering in Short Read Mapping,” *arXiv preprint arXiv:2103.14978*, 2021.

