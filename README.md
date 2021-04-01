# GateKeeper-GPU: Pre-Alignment Filtering in Short Read Mapping
Calculating the similarities and differences between two sequences is a computationally expensive task since approximate string matching techniques traditionally inherit dynamic programming algorithms with quadratic time and space complexity. We introduce GateKeeper-GPU, a fast and accurate pre-alignment filter that efficiently reduces the need for expensive sequence alignment. GateKeeper-GPU improves the filtering accuracy of GateKeeper (Alser et al.), and by exploiting the massive parallelism provided by GPU threads it concurrently examines numerous sequence pairs rapidly. 

## Usage
### Compilation
In preprocessing steps of GateKeeper-GPU, the sequences are encoded in 2-bit format. We provide GateKeeper-GPU in two versions: host-encoded and device-encoded.  Encoding is carried out in CPU in host-encoded version and the bulk of encoded sequences are transfered to GPU. In device-encoded version, sequences are copied  to GPU in string format and each GPU thread encodes its own sequence pairs. Depending on the read mapper's workflow, the most advantageous version can be utilized. If the mapper already has an encoding mechanism, using host-encoded version would be more suitable. 


## Citation
Please visit our full paper arXiv preprint at [GateKeeper-GPU](https://arxiv.org/abs/2103.14978). 
You can cite it as:
>  Z. Bing ̈ol, M. Alser, O. Mutlu, O. Ozturk, and C. Alkan, “Gatekeeper-GPU: Fast and Accurate Pre-alignment Filtering in Short Read Mapping,” 2021.

