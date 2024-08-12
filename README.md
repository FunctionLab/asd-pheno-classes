# Decomposition of phenotypic heterogeneity in autism reveals distinct and coherent genetic programs

## Contents

- [Overview](#Overview)
- [System requirements](#system-requirements)
- [Data Availability Statement](#data-availability-statement)


# Overview

Unraveling the phenotypic and genetic complexity of autism is extremely challenging yet critical for understanding the biology, inheritance, trajectory, and clinical manifestations of the many forms of the condition. Here, we leveraged broad phenotypic data from a large cohort with matched genetics to characterize classes of autism and their patterns of core, associated, and co-occurring traits, ultimately demonstrating that phenotypic patterns are associated with distinct genetic and molecular programs. We used a generative mixture modeling approach to identify robust, clinically-relevant classes of autism which we validate and replicate in a large independent cohort. We link the phenotypic findings to distinct patterns of de novo and inherited variation which emerge from the deconvolution of these genetic signals, and demonstrate that class-specific common variant scores strongly align with clinical outcomes. We further provide insights into the distinct biological pathways and processes disrupted by the sets of mutations in each class. Remarkably, we discover class-specific differences in the developmental timing of genes that are dysregulated, and these temporal patterns correspond to clinical milestone and outcome differences between the classes. These analyses embrace the phenotypic complexity of children with autism, unraveling genetic and molecular programs underlying their heterogeneity and suggesting specific biological dysregulation patterns and mechanistic hypotheses.

# **System Requirements** 

## Hardware requirements

The code in this repository requires only a standard computer with enough RAM to support the operations defined in the scripts. Only a CPU is needed to run all code in this repository.

## Software requirements

The conda environment used in the testing of all scripts is described in `conda_requirements.txt`, which contains all software dependencies, including version numbers.

We recommend starting a new conda environment using the requirements file:

```
conda create --name asd_env --file conda_requirements.txt
conda activate asd_env
conda list # verify the installation
```

### **Data Availability Statement**

In order to abide by the informed consents that individuals with autism and their family members signed when agreeing to participate in a SFARI cohort (SSC and SPARK), researchers must be approved by SFARI Base (https://base.sfari.org).
