FEniCS
======
In order to be able to run this code you need to have FEniCS installed.
The best way to achieve this is to use a `dockerized` installation to run
FEniCS. Refer to

`FEniCS installation guide <https://fenics.readthedocs.io/projects/containers/en/latest/introduction.html#installing-docker>`_

FEniCS installation usually includes a minimum set of python libraries.
However, you might need to install additional ones like.

It is recommended to create a named container with a folder shared with the
local os:

.. code-block:: console

   $ docker run -ti -v $(pwd):/home/fenics/shared --name fenics-container quay.io/fenicsproject/stable

To start the container run

.. code-block:: console

   $ docker start fenics-container

To stop the container run

.. code-block:: console

   $ docker stop fenics-container


To run the container we can create a shell script containing

.. code-block:: shell
   :caption: run_fenics.sh

   #!/bin/bash
   docker exec -ti -u fenics fenics-container /bin/bash -l

Add execution permissions to the script

.. code-block:: console

   $ chmod +x run_fenics.sh

Then, we can just access the container by

.. code-block:: console

   $ ./run_fenics.sh


Python Modules
==============
To run the analysis on the client side, make sure you have the following packages

1. Matplotlib
2. Scipy
3. h5py
4. pandas
5. tqdm

Installation of dependencies using PIP
--------------------------------------

Install the matplotlib package

.. code-block:: console

   $ pip install matplotlib

Install scipy

.. code-block:: console

   $ pip install scipy

Install the h5py package

.. code-block:: console

   $ pip install h5py

Install pandas

.. code-block:: console

   $ pip install pandas

Install tqdm (for progress bars)

.. code-block:: console

   $ pip install tqdm

Installation of dependencies using conda
----------------------------------------
Conda distributions usually come with matplotlib, scipy. In case your distribution does not include it you can run

.. code-block:: console

    $ conda install matplotlib
    $ conda install scipy


Install the h5py package

.. code-block:: console

   $ conda install h5py

Install pandas

.. code-block:: console

   $ conda install pandas

Install tqdm (for progress bars)

.. code-block:: console

   $ conda install -c conda-forge tqdm