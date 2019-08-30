#### ACSE-9 Independent Research Project

# Implementing integrated machine learning strategies to accelerate high accuracy fracture growth simulators 

### Files

The only files directly relevant to examination are in the top-level directory, namely Report.pdf containing my report, Documentation_UserManual.pdf containing the Documentation and User Manual, and finally solution.py which is the code.

Inside the Data folder you will find the base CSVs used in training. The raw data as given to me is too large to upload, so only the cleaned CSVs are present.

The Figures folder contains figures for both the final report and the project plan.

The Models folder contains all the models used to provide the graphs for the testing and results sections in the report. Feel free to use the code and load the models to verify the outputs.

Papers contains a number of papers earmarked for usage in the project, although not all were used and others not present were.

Prototypes holds the previous incarnations of the code. These programs were not maintained, so they will almost certainly be non-functional.

The Project_Plan folder keeps files related to the original project plan.

Finally, the Report folder contains the report draft, as well as some other notes and designs which eventually didn't come to fruition.

### Code Operation

See the documentation and user manual for further information. In short, beyond the dependencies listed in the documentation, the project will run straight away. Getting it to do certain things just relies on setting the flags on main.

#### Input
The code takes as input a CSV data file, which does not possess an ID column nor a header column.
It should be structured as (see documentation for further detail):

  x | y | z | nx | ny | nz | ux | uy | uz | KI | KII | KIII | angle | target KI | target KII | target KIII | target angle
  
If loading a model, further information must be provided as the ld_mdl argument, see documentation for details.

#### Output
When set to save mode, the output is a .pt file containing the state dict of the trained model.
If it is not saving, there is no permanent output, beyond the graphs you may tell it to generate.
