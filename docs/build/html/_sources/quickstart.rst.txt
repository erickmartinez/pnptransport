Quickstart: Finite Source Simulations
=====================================
The simplest way to run the code is to run a simulation using an ini file from
the command line.

Shell scripts are available at the `executables` folder in the root of
the installation. If they do not already have execution permissions run:

.. code-block:: console

   $ cd executables
   $ chmod +x *.sh

Running a finite source simulation. From the root of pnptransport run

.. code-block:: console

   $ ./simulate_fs.py --config input_example.ini

where the .ini file looks like

.. literalinclude:: ./input_example.ini
   :linenos:
   :caption: input_example.ini
   :language: ini
   :name: Finite source example

The sections of the input file are

global
------
This section contains parameters that are not layer specific including

**simulation_time**: str
   This corresponds to the total time to be simulated in seconds.
**temperature**: float
   The simulated temperature in °C. Used to determine ionic mobility
   in the dielectric, according to :math:`\mu = D q / k_{\mathrm{B}} T`.
**surface_concentration**: float
   The surface concentration at the source :math:`S`, given in
   cm\ :sup:`-2` \. Used to determine the flux at the source, given
   by :math:`J_{\mathrm{0}} = k S`, where :math:`k` is the rate of ingress.
**rate_source**: float
   The rate of ingress of ionic contamination at the source, in s\:sup:`-1`\.
   Used to determine the flux at the source, :math:`J_{\mathrm{0}} = k S`.
**filetag**: str
   The file tag used to generate the output folder and files.
**time_steps**: int
   The number of time intervals to simulate.
**h**: float
   The surface mass transfer coefficient in cm/s, for the segregation flux
   at the SiN\ :sub:`x` \ / Si interface.
**m**: float
   The segregation coefficient at the dielectric/semiconductor interface.
**recovery_time**: float
   The additional simulation time in seconds without PID stress used for
   recovery.
**recovery_voltage**: float
   The bias used during recovery in V. This is applied to the dielectric
   layer and ideally needs to be negative.
**cb**: float
   The background concentration in cm\ :sup:`-3` \. Used as a finite initial
   concentration.
**er**: float
   The relative permittivity of the dielectric.

sinx
----
**d**: float
   The diffusion coefficient of the ionic species in the dielectric
   in cm\ :sup:`2` \/s.
**stress_voltage**: float
   The applied voltage stress in the film in V.
**thickness**: float
   The thickness of the layer in um.
**npoints**: int
   The number of grid points to simulate.

si
----
**d**: float
   The diffusion coefficient of the ionic species in the semiconductor
   in cm\ :sup:`2` \/s.
**stress_voltage**: float
   The applied voltage stress in the film in V.
**thickness**: float
   The thickness of the layer in um.
**npoints**: int
   The number of grid points to simulate.


The code needs to be run in a linux terminal. However, it is recommended to
use a graphical environment to keep the processes alive if the remote
connection with the server fails.

The default directory structure of the simulation will be

::

   top_folder
   |---base_folder
   |   |---input_file.ini
   |---results
   |   |---constant-flux
   |   |   |---filetag.h5
   |   |   |---filetag.ini


The results folder can be specified by the optional argument `--output` to
the `simulate_fs.py` script

.. code-block:: console

   ./simulate_fs.py --config input_file.ini --output folder_output

which will generate a folder structure like this.

::

   top_folder
   |---base_folder
   |   |---input_file.ini
   |---output_folder
   |   |---filetag.h5
   |   |---filetag.ini