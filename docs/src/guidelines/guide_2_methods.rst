#################################
Contribution's guideline: Methods
#################################

Thank you very much for considering contributing to the methods collection of **NetworkCommons**! For methods, it is especially important that inputs and outputs are 
compatible with the rest of the package, the purpose is stated and the assumptions of the method are clear.


----------------
1. Documentation
----------------

In the :doc:`./docs/src/methods.rst file <../methods>`, contributors should add:

* The description of the method
* A figure showcasing the basics (if possible)
* Input/output definition
* Link to publication and repository (if available)

For example:

.. literalinclude:: ../methods.rst
   :language: rest
   :lines: 14-30

Functions should be documented using `Google style Python docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

In :doc:`./docs/src/api.rst file <../api>`, contributors should add a new documentation module that contains the new classes/functions implemented:

.. literalinclude:: ../api.rst
   :language: rest
   :lines: 40-53
    
------
2. API
------

* Every new method should be implemented in a separate file (e.g `_moon.py`) inside `/networkcommons/methods/`. 
* Contributors can then implement their own set of functionalities and expose those necessary to the public API via the `__all__` variable (see other files for examples).
* The input of the overall pipeline must be at least a `Network` object, and its overall output should return at least a `Network` object containing the contextualised network. 
This does not apply to intermediate functions (e.g `Network` --function 1--> `pd.DataFrame` --function 2--> `Network`) in case of a pipeline containing several functions, such as MOON.