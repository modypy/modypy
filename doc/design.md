# Design Documentation

## Mathematical Model of Blocks

Each *block* implements a dynamical system of the general form

    x(t) = x0
    dx/dt = g(t,x(t),u(t))
    y(t) = h(t,x(t),u(t))

with the state vector `x(t)`, the input vector `u(t)` and the output vector `y(t)`.

If `y(t)` directly depends on component `ui(t)` of `u(t)`, then `ui(t)` is said to be a *feed-through* input.

## Composition of Block Graphs

A block graph is a tree structure, with a single root, and blocks as nodes.
The leaves of the tree are leaf blocks, all other nodes of the tree are non-leaf blocks.
If a block instance is contained multiple times in the same tree, the results are undefined.

The non-leaf blocks represent the composition of their child blocks and do not contain individual dynamics.
Instead, their inputs and outputs are connected to inputs and outputs of the child blocks.
In addition, outputs of child blocks may be connected to inputs of child blocks.

## Compilation

During compilation, a block graph is compiled into a structure consisting of

 - information about the state and output vectors of the dynamical system represented by the block graph,
 - a mapping of the states and outputs of the leaf blocks into the state and output vectors,
 - a mapping of the inputs and outputs of all blocks into the output vector,
 - an execution sequence of leaf blocks honoring the inter-dependencies of leaf blocks represented by the input-output connections and feed-through information.

### The State and Output Vectors

The state and output vectors of a dynamical system represented by a block graph are composed of the state and output vectors of the leaf blocks contained in the graph.
Non-leaf blocks do not have state, and their inputs and outputs are merely symbolic in nature, connecting inputs and outputs of leaf blocks across non-leaf block borders.

### Input Mapping

Each input of a block may be connected to at most one output of another block.
During compilation, each input is mapped to an entry of the output vector that represents the output the input is connected to.
During this process, each output of each non-leaf block is also mapped to the entry of the output vector.

### Execution Order

The derivatives of the state vectors of individual leaf blocks may depend on the current value of both the state vector and the inputs of that leaf block.
Thus, before calculating the state derivative, the input values need to be determined, which in turn are taken from the outputs of possibly other blocks.

However, the outputs of a block may in turn depend on the inputs of that block.
Therefore, the output vector of the whole system needs to be filled in an order that is compatible with these dependencies.

Such an execution order is determined during compilation using the information about inter-connections and applying Kahn's algorithm to this information.

## Principal Operation of the Main Simulation Loop

1. Compilation
    1. Collect all leaf blocks in the graph.
    2. Establish global state, output and event vectors, mapping the states, outputs and events of each block onto a contiguous sequence of entries in the respective global vectors.
    3. For each input, determine the source index in the global output vector.
    4. Determine the execution order of the blocks in the graph.
2. Initialization
    1. Initialize the current time to the start time of integration.
    2. Initialize the state vectors from the definitions in the non-virtual blocks in the graph.
    3. Initialize the global output vector by evaluating the output function for each non-virtual block according to the execution order.
    4. Initialize the global event function vector by evaluating the event function for each non-virtual block according to the execution order.
    5. Add the current time, state, output and event vector to the simulation result.
3. Simulation
   1. Integration (repeated until the current time equals the final time)
      1. Determine the next time step
         1. If integration times are specified explicitly: Use the minimum of t+dt and the next integration time specified.
         2. Otherwise use the minimum of t+dt and the end time of integration.
      2. Integrate until the next time step
      3. Event handling
         1. If an event occurred (event function has changed sign), determine the time of the event for each event having occurred.
         2. Determine the time of the first event having occurred.
         3. Re-integrate only until the time of the event.
      4. Add the current time, state, output and event vector to the simulation result.
   2. Return the simulation result.

## Implementation Details

### Mapping of inputs to the output vector

The input map is a mapping of block inputs to elements of the output vector.

For each block B and each output i of B, let `outmap[B,i]` be the index of the respective signal in the output vector.
For each block B and each input i of B, let `inmap[B,i]` be the index of the respective signal in the output vector.

We assume that `outmap[B,i]` is pre-populated for leaf blocks B.
We further assume that `inmap` is pre-populated to a value marking each entry as undefined.

1. For each non-leaf block B in the graph, traversed in post-order: For each connection from output i of internal block C to the output j of B, let `outmap[C,j]:=outmap[B,i]`.
2. For each internal connection from output i of block B to input j of block C, let `inmap[C,j]=outmap[B,i]`.
3. For each non-leaf block B in the graph, traversed in pre-order: For each connection from input i of B to input j of the block C, let `inmap[C,j]:=inmap[B,i]`.

After this, `inmap[B,i]` is either undefined or contains the index of the output connected to input i of block B in the output vector.
If it is undefined, the input i of block B is not connected to any output.

### Kahn's Algorithm

Input:
   - Let `N' be the number of nodes.
   - Let `num_incoming(n)` be the number of incoming edges for node `n`, with `n=0,...,N-1`.
   - Let `num_outgoing(n)` be the number of outgoing edges for node `n`, with `n=0,...,N-1`.
   - Let `dest(n,i)` be the target node of edge i from node `n`, with `n=0,...N-1` and `i=0,...,outedges(n)-1`.

1. Let `L` be the empty sequence
2. Let `S:={n|n=0,...,N-1 and incoming(n)==0}'.
3. While `S` is not empty
   1. Remove an element `src` from `S`
   2. Add `src` to `L`
   3. For `i=0,...,num_outgoing(src)-1':
      1. Let `tgt:=dest(src,i)`
      2. Let `num_incoming(tgt):=num_incoming(tgt)-1`
      3. If `num_incoming(tgt)` is zero, add `tgt` to `S`

If there is an `n=0,...,N-1` with `incoming(n)>0` after this, the dependency graph was not acyclic.

Otherwise, `L` is a sequence of nodes where node `i` is guaranteed to precede `j` if there was an edge from `i` to `j` in the original graph.

# Simulation Architecture

1. The system graph is built of blocks.
2. Blocks can be either leaf or non-leaf blocks.
3. Blocks shall provide their number of inputs and outputs as the properties `num_inputs` and `num_outputs`.
4. Non-leaf blocks shall provide a method `enumerate_leaf_blocks`, yielding all the leaf blocks they contain directly or indirectly.
5. Non-leaf blocks shall provide the list of immediate child blocks they contain as a property `children`.
5. Leaf blocks shall provide the number of dimensions of their state vector as property `num_states`.
6. Leaf blocks shall provide a method `output_function` accepting the time and the current state as input and providing the output vector as result.
7. Leaf blocks shall provide a method `state_update_function` accepting the time, the current state and the current input as inputs and providing the time derivative of the state vector as output.
8. Leaf blocks shall provide the list of inputs that influence the output directly by the property `feedthrough_inputs`.
9. Non-leaf blocks shall provide a method `enumerate_internal_connections`, yielding all internal connections, giving a tuple of source block, source output index, destination block and destination input index.
10. Non-leaf blocks shall provide a method `enumerate_input_connections`, yielding all connections from an input of the containing block, giving a tuple of external input index, destination block and destination input index.
11. Non-leaf blocks shall provide a method `enumerate_output_connections`, yielding all connections from an input of the containing block, giving a tuple of source block, source output index and external output index.
