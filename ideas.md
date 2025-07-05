## Design for fully incremental computation:
- Accumulator at every node
- Every node outputs a non-zeroed value and a zero switch
- This means we can keep track of a continuous updateable value and turn on zero switch when zero happens rather than throwing away our increment
- MAJOR WORLD SHATTERING PROBLEM: requires 1 input buffer + zero switch for every connection into a neuron, which is basically just fully-unrolled inference
- So the compromise is: 2nd layer inference (1st layer of actual computation, "hidden layer" inference) is incremental, maintaining incremental outputs 

## Investigate if this technique could be called "candidate elimination"


# Fun extensions

## Add an abstract domain tracker to simulator, to determine how good the simulated-hardware-calculated domain actually is
For networks made with only matmul, bias and ReLU, 'abstract' can simply mean a polytope which is relatively easy to keep track of.
Open question as to how to communicate this information - perhaps simply the ratio of the volume of the abstract domain to the volume of the hardware-calculated domain

## Come up with a nice algorithm to determine exact polytope bounds through layers with uncertainty in weights etc
Required for monte carlo evaluation of hyperrectangle bounds used in hardware, compared to ideally achievable bounds

## Keep track of a statistical distribution of the domain of the result instead of hard bounds

## Use more-abstract-domain bounds (octohedral etc.)
Mitigates problem of non-relational n-cuboid domain growing vastly bigger than the polytope it bounds
Suggested by Ka Wing Li

## Process fragments as 2-bit exponent + sign 
Is there any possible advantage to this?

## Determine whether middle of input interval ends up (on average) as middle of output interval
Would allow keeping track of just the interval, not an extra middle point

## Consider constructing MLE (maximum likeliness estimator) or similar instead of midpoint
Or any other metric? Might be value in lower, middle, upper bounds telling us something definitive about output distribution

# Sample application
Weight bits are received one-by-one, in parallel (i.e. MSB of all inputs received, then 2nd-MSB of all inputs received...), and are converted to centroid-interval representation progressively