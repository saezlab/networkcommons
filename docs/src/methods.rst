#######
Methods
#######

Here you can find a collection of the methods implemented in **NetworkCommons**, along with detailed descriptions.
If you want to contribute with your own, please check our :doc:`Contribution guidelines <guidelines/guide_2_methods>`.

-------------------
Topological methods
-------------------

.. _details-sp:

Shortest path
-------------

The shortest path is an algorithm for finding one or multiple paths that minimize the distance from a set of starting nodes to a set of destination nodes in a weighted graph (https://doi.org/10.1007/BF01386390).

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_sp.svg" width="1000px"></object>

   
**Input:** Set of source and target nodes, (weighted) network graph

**Node weights:** w(v) = 1

**Edge weights:** 0 ≤ w(e) ≤ 1

**Functions:** See API documentation for :ref:`Topological methods <api-topological>`.


Sign consistency
----------------

The sign consistency method checks for sign consistency between the nodes in a given graph. Hereby, source and target nodes, as well as the edges in the graph have an assigned sign. 

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_sign.svg" width="1000px"></object>


**Input:** Set of source and target nodes (with a sign for up- or downregulation), network graph

**Node weights:** w(v) ∈ {1, −1}

**Edge weights:** w(e) ∈ {1, −1}

**Functions:** See API documentation for :ref:`Topological methods <api-topological>`.


Reachability filter
-------------------

The reachability filter generates a network consisting of all reachable nodes from a set of starting nodes.

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_reach.svg" width="1000px"></object>


**Input:** Set of source nodes, network graph

**Node weights:** w(v) ∈ {1}

**Edge weights:** w(e) ∈ {1}

**Functions:** See API documentation for :ref:`Topological methods <api-topological>`.


All paths
---------

All paths find all possible connections between a set of source nodes and a set of target nodes. In contrast to the shortest path method or the sign consistency method it doesn’t take the distance or any sign information into account, respectively.

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_ap.svg" width="1000px"></object>


**Input:** Set of source and target nodes, network graph

**Node weights:** w(v) ∈ {1}

**Edge weights:** w(e) ∈ {1}

**Functions:** See API documentation for :ref:`Topological methods <api-topological>`.

--------------------------------------
Random walk with restart (RWR) methods
--------------------------------------

Page rank
---------

The Page rank algorithm initially calculates a weight for each node in a graph based on a random walk with restart method. It starts at a set of source or target nodes and determines the importance of the other nodes in the graph based on the structure of the incoming or outgoing edges. It then builds a network considering the highest-ranking nodes starting from each of the source and the target nodes.

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_ppr.svg" width="1000px" alt="PPR"></object>


**Input:** Set of source and target nodes, network graph

**Node weights:** w(v) ∈ {1}

**Edge weights:** w(e) ∈ {1}

**Functions:** See API documentation for :ref:`RWR methods <api-rwr>`.

----------------------------
Recursive enrichment methods
----------------------------

MOON
----

MOON (meta-footprint method) performs iterative footprint activity scoring and network diffusion from a set of target nodes to generate a sign consistent network (https://doi.org/10.1101/2024.07.15.603538). Starting from a set of weighted target nodes it calculates a weight for the next layer of upstream nodes using a univariate linear model. This process is repeated until a set of source nodes or a certain number of steps is reached. Hereby, any source node with an incoherent sign between MOON and the input sign is pruned out along with all incoming and outgoing edges. Additionally, edges between two inconsistent nodes are removed.


**Input:** Set of weighted target nodes (and optionally weighted source nodes), network graph

**Node weights:** w(v) ∈ ℝ

**Edge weights:** w(e) ∈ ℝ

**Functions:** See API documentation for :ref:`MOON <api-moon>`.

Preprocessing
~~~~~~~~~~~~~

The MOON scoring system starts by removing self-interactions and interactions with non-defined signs (neither +1 nor -1). Then, we must add compartimental information to the metabolic measurements that will be used as downstream measurements. 
Then, we filter out those inputs that cannot be mapped to the prior knowledge network.

Network compression
~~~~~~~~~~~~~~~~~~~

This is one of the most important parts of this vignette. Here, we aim to remove redundant information from the network, in order to reduce its size without compromising the information contained in it. A common example would be the following:

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_moon_comp_normal.svg" width="1000px" alt="MOON"></object>


However, in other cases, we would lose information:

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_moon_comp_edgecases.svg" width="1000px" alt="MOON"></object>


MOON scoring
~~~~~~~~~~~~

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_moon_core.svg" width="1000px" alt="MOON"></object>


Network decompression and solution network obtention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To finish, the compressed nodes are restored to their original state, and the solution network is obtained by establishing a threshold for the MOON scores of the nodes.
In addition, users can rename the nodes to human-readable names.

-----------------
ILP-based methods
-----------------

CORNETO - CARNIVAL
------------------

CORNETO (Constraint-based Optimization for the Reconstruction of NETworks from Omics) is a unified network inference method which combines a wide range of network methods including CARNIVAL which is currently implemented in NetworkCommons. CARNIVAL (CAusal Reasoning for Network identification using Integer VALue programming) connects a set of weighted target and source nodes using integer linear programming (ILP) and predicts the sign for the intermediate nodes (https://doi.org/10.1038/s41540-019-0118-z). Thereby, it optimizes a cost function that penalizes the inclusion of edges as well as the removal of target and source nodes. Additionally, it considers a set of constraints that among other things do not allow sign inconsistency.

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_corneto.svg" width="1000px" alt="MOON"></object>

**Input:** Set of weighted target and source nodes, network graph

**Node weights:** w(v) ∈ ℝ

**Edge weights:** w(e) ∈ {1, −1}

**Functions:** See API documentation for :ref:`CORNETO <api-corneto>`.

SignalingProfiler
------------------

SignalingProfiler (https://doi.org/10.1038/s41540-024-00417-6) Python implementation is a two-steps pipeline.
In the first step, SignalingProfiler generates the Naïve Network, a hierarchical and multi-layered network between source and target nodes using networkcommons "All paths".
Three different layouts can be chosen, defined as one-, two-, or three-layers networks, with an increasing level of deepness.

Each layer is defined by a different set of molecular functions.
The molecular function for each protein is obtained by parsing the UNIPROT database GO Molecular Function annotation according to relative GO Ancestor terms. 
This molecular function annotation refers to signal trasduction context: K, kinase; PP, phosphatases; T, transcription factor; O, all the other molecular functions.

In the one-layer network, the perturbed node is connected to every target and is molecular function agnostic.
The two-layers network connects the perturbed node to kinases/phosphatases/others (first layer) and then connect the latters to transcription factors (second layer). 
The three-layers network adds another layer between kinases/phosphatases and other signaling proteins.

In the second step, SignalingProfiler calls "CORNETO - CARNIVAL" to retrieve only sign-consistent edges from the naïve network (removing grey dashed edges).

.. raw:: html

   <object type="image/svg+xml" data="_static/nc_signalingprofiler.svg" width="1000px" alt="SignalingProfiler"></object>

**Input:** Set of weighted target and source nodes, network graph

**Node weights:** w(v) ∈ ℝ

**Edge weights:** w(e) ∈ {1, −1}

**Functions:** See API documentation for :ref:`SignalingProfiler <api-signalingprofiler>`.