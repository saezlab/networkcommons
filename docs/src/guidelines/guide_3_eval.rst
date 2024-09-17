###############################################
Contribution's guideline: Evaluation strategies
###############################################

Thank you for considering contributing to the evaluation strategies. To implement a benchmark strategy, we need to clearly state the goals 
and assumptions behind the strategy, define suitable datasets and define one (or more) performance metrics. For other examples, see other :doc:`Evaluation strategies <../benchmarks>`.

----------------
1. Documentation
----------------

Each new benchmark strategy should inform of the following points:

* **Data:** which types of data/scenarios can be used for this strategy.
* **Assumption:** this is the most important part. Here, we define the idea behind the strategy, you can think of it as a small workflow draft.
* **Performance metric:** which metric we will use to rank the methods.
* **A note block**: here, contributors can explain in a nutshell how this evaluation metric can "differentiate" good and bad performers. It acts as a summary of the aforementioned points. 

For example:

.. literalinclude:: ../benchmarks.rst
   :language: rest
   :lines: 85-108


------
2. API
------

* New strategies can be included in a separate file (e.g ``_eval1.py``) inside the ``networkcommons.eval`` module. 
* Contributors can then implement their own set of functionalities and expose those necessary to the public API via the ``__all__`` variable (see other files for examples).
* The input must be at least a ``Network`` or dict of ``Network`` objects (``{'name1': Network1, 'name2': Network2, ...}``). The output can be anything, but ideally a ``pandas.DataFrame``,
with columns 'network' containing the network ID or name, and a number of columns from the implemented metric(s).




