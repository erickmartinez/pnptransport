.. PNP Transport documentation master file, created by
   sphinx-quickstart on Thu Aug 13 10:17:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PNP Transport's documentation!
=========================================
This framework uses FEniCS to estimate the numerical solution to
Poisson-Nernst-Planck equation to solve the transport kinetics
of charged species in dielectrics and stacks of materials.

Dependencies
============

.. toctree::
   :maxdepth: 2

   dependencies


Quick Start
===========

.. code-block:: console

   $ cd executables
   $ chmod +x *.sh

Running a finite source simulation

.. code-block:: console

   $ cd ./executables
   $ ./simulate_fs.py --config input_example.ini

where the .ini file looks like

.. literalinclude:: ./input_example.ini
   :linenos:
   :caption: input_example.ini
   :language: ini

.. toctree::
   :maxdepth: 1

   quickstart



.. toctree::
   :maxdepth: 2
   :caption: Contents:


Batch Analysis
==============
.. toctree::
   :maxdepth: 4

   ofat

FEniCS Transport Simulations Code
=================================

.. toctree::
   :maxdepth: 4

   pnptransport

PID Simulation Module
=================================

.. toctree::
   :maxdepth: 4

   pidsim

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
