# Network Neuroethology

Tools of analysis: Network Neuroethology Analysis (in collaboration with Dr. Madhavun Candadai)


Computational neuroethology approach to uncover what a statistical measure of functional connectivity tell us about the actual functional connectivity of a nervous system. 

In order to address this theoretical challenge, we propose the following paradigm. In scenario (*A*), the subject is presented with two different tasks (blue and magenta). For each task, there are multiple trials (different sizes, different starting conditions). For each trial, neural activity is recorded for the subject. From the combined neural recordings of each task, a node functional connectivity (nFC) is created, using one of three techniques: Pearson's correlation, mutual information, and transfer entropy. Finally, from the nFC the subcircuit for each task is estimated. 

In scenario (*B*), the same subject is now tested on the same two tasks, but now the drop in performance is recorded during information lesions to each pair of connections or each individual connection between the components of the subject's brain. This effect of lesions per pairwise component is considered the actual functional connectivity (aFC). From it, the ground truth functional circuit is obtained for each task, which is used to assess the usefulness of the statistical nFC approach.  We cannot do part B of this approach with humans, or with any  other living organism, given current experimental limitations and ethical considerations. However, we can use artificial life techniques to: first generate agents capable of multiple tasks, and then analyze them in the way proposed above.

