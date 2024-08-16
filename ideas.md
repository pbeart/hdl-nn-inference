## Add an abstract domain tracker to simulator, to determine how good the simulated-hardware-calculated domain actually is
For networks made with only matmul, bias and ReLU, 'abstract' can simply mean a polytope which is relatively easy to keep track of.
Open question as to how to communicate this information - perhaps simply the ratio of the volume of the abstract domain to the volume of the hardware-calculated domain

## Keep track of a statistical distribution of the domain of the result instead of hard bounds

## Use more-abstract-domain bounds (octohedral etc.)
Mitigates problem of non-relational n-cuboid domain growing vastly bigger than the polytope it bounds
Suggested by Ka Wing Li

## Process fragments as 2-bit exponent + sign 
Is there any possible advantage to this?
