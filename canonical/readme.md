# Changes
- remove sat names from commander's intent; no reason to have different specs for different satellites
- removed `displayStruct` for the comms network; just always show it
- invert dependency between `comms` and `satellites`; a satellite generic access to the comms network, whereas the network manages the simulated comms and is agnostic to what it's nodes represent
- switch to agent names in the comms network; a sat can reason about
- `*_initialize` is no longer an estimate method. The estimator manages whether or not it needs to be initialized
- `predict` invokes `initialize` as needed; this means `update` is always called with the initial measurement (previously the first measurement was skipped in favor of `initialize`)
