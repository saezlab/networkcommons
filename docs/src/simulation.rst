Cyclic dynamical simulation
===========================

``networkcommons.simulation`` generates continuous real-state data with cycles,
interventions, and optional independent Gaussian process noise. Negative states
are allowed. This is a declared dynamical family, not a simulator for every
structural causal model. The dynamics use the Itô convention and unit decay.

Model and interventions
-----------------------

The interaction matrix uses receiver rows and source columns: ``A[i, j]``
represents the influence of node ``j`` on node ``i``. Production is

.. math::

   F(x) = q + p \odot \phi(Ax+b), \qquad
   dX = [F(X)-X]dt + \operatorname{diag}(\sigma)dW.

Basal input ``b`` acts before the transfer. Production offset ``q`` acts after
it. ``Transfer`` supports identity, tanh, thresholded tanh with a linear
component, and saturation centered at a specified value with unit local slope.
All model arrays and mappings are copied and read-only.

.. code-block:: python

   import numpy as np
   from networkcommons.simulation import (
       CyclicSystem, Transfer, intervene, simulate_dynamics, observe,
       solve_equilibrium, sample_linear_stationary,
   )

   A = np.array([[0., .2, 0.], [0., 0., .3], [.1, 0., 0.]])
   center = np.array([.2, .3, .4])
   system = CyclicSystem(
       A, basal = center - A @ center,
       transfer = tuple(
           Transfer.centered_tanh(center = c, curvature = 8.)
           for c in center
       ),
   )
   condition = intervene(system, clamp = {0: 0.})
   run = simulate_dynamics(
       condition, initial_state = center,
       dt = .01, n_steps = 10000, process_seed = 17,
   )
   observed = observe(
       np.repeat(run.final_states, 250, axis = 0),
       operator = np.eye(3)[[1, 2]], noise_sd = .01,
       measurement_seed = 31,
   )

The clamp replaces node zero's equation and fixes its latent state exactly.
Its outgoing interactions remain active. Receiver edits on a resulting clamp
raise an error. An outgoing-scale edit on a clamped source is allowed.

``input_shift`` adds to basal inputs. ``incoming_scale`` multiplies receiver
rows and ``outgoing_scale`` multiplies source columns. Transfer, production,
and diffusion edits set effective values. Each call returns a new model.
Build alternative experiment conditions directly from the baseline.

Partial production replacement with fraction ``s`` and level ``v`` sets
``p_new = (1-s)*p`` and ``q_new = (1-s)*q+s*v``. Set diffusion separately.
Constant production with nonzero diffusion remains noisy and differs from an
exact state clamp. There is no implicit intervention-strength flag.

Choose the output law explicitly
--------------------------------

* ``simulate_dynamics`` implements Euler–Maruyama, or Euler when diffusion is
  zero. It returns finite-time states. Supply one initial row per independent
  path. Saved trace times from one path are correlated observations.
* ``solve_equilibrium`` solves structural equations. Fixed-point iteration
  requires an initial vector and checks equation residuals. ``method='linear'``
  uses a direct solve when active free mechanisms are affine. An equilibrium
  can exist even when the dynamics are unstable.
* ``sample_linear_stationary`` draws independent Gaussian samples from an
  attracting affine stationary law. It checks the reduced intervened operator,
  solves a continuous Lyapunov equation, and reports numerical diagnostics.
  Degenerate diffusion requires no diagonal jitter.

Finite-time integration does not certify stationarity. A nonlinear stochastic
population mean generally differs from a deterministic equilibrium. Assess
approximate stationary sampling through longer durations, smaller time steps,
and independent-path distribution summaries with Monte Carlo uncertainty.

.. code-block:: python

   linear = CyclicSystem(A, basal = np.ones(3), diffusion = .1)
   equilibrium = solve_equilibrium(linear, method = 'linear')
   stationary = sample_linear_stationary(
       linear, n_samples = 250, process_seed = 29,
   )

   # Independent randomness for latent paths and measurement noise.
   process_child, measurement_child = np.random.SeedSequence(42).spawn(2)
   process_seed = int(process_child.generate_state(1, dtype = np.uint64)[0])
   measurement_seed = int(
       measurement_child.generate_state(1, dtype = np.uint64)[0]
   )
   paths = simulate_dynamics(
       linear, initial_state = np.zeros((250, 3)),
       dt = .01, n_steps = 2000, process_seed = process_seed,
   )
   measurements = observe(
       paths.final_states, noise_sd = .01,
       measurement_seed = measurement_seed,
   )

The independent-stream construction matters: reusing the same integer for both
seeds does not create independent streams. Local generators use PCG64.
Saving traces consumes no additional process draws. Reproducibility requires
compatible numerical libraries as well as matching parameters and seeds.

Validation and dataset examples
-------------------------------

Run the example from a NetworkCommons checkout:

.. code-block:: console

   uv run --locked python examples/cyclic_simulator.py

It writes five NPZ/JSON pairs under ``build/simulation``:

* Deterministic linear and saturating equilibria, with shifts and hard clamps.
* Linear stochastic distributions with intervention-specific production and noise.
* Coupling/saturation families sharing a heterogeneous control state.
* A fixed-duration process-noise sweep including zero diffusion.
* Partial production replacements and a separately labeled hard clamp.

Each record includes effective parameters, transfer definitions, initial states
when used, numerical settings, output law, seed derivation, readout, variable
order, condition splits, versions, and source hashes. NPZ files contain numerical
and string arrays and require no pickle loading. Dataset assembly and split
choices stay outside the numerical core. Clean reference responses must not
enter fitting or model selection.

The unit tests compare scalar finite-time moments with both the Euler law and
the continuous-time law. They refine the time step at fixed duration, then
increase duration separately. Coupled stationary tests check the Lyapunov
equation, empirical means and covariance, and exact target covariance under clamps.

API reference
-------------

.. automodule:: networkcommons.simulation
   :members:
   :imported-members:
