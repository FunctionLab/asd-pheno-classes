# Decomposition of phenotypic heterogeneity in autism reveals underlying genetic programs

## Repo Contents

- [Preprocessing](https://github.com/FunctionLab/asd-pheno-classes/tree/main/PreprocessingScripts)
- [Constructing Phenotype Classes](https://github.com/FunctionLab/asd-pheno-classes/tree/main/PhenotypeClasses)
- [Phenotype Validation and Replication](https://github.com/FunctionLab/asd-pheno-classes/tree/main/PhenotypeValidations)
- [Rare Variant Genomic Analyses](https://github.com/FunctionLab/asd-pheno-classes/tree/main/GenomicAnalyses)


# Overview

Unraveling the phenotypic and genetic complexity of autism is extremely challenging yet critical for understanding the biology, inheritance, trajectory, and clinical manifestations of the many forms of the condition. Using a generative mixture modeling approach, we leveraged broad phenotypic data from a large cohort with matched genetics to identify robust, clinically-relevant classes of autism and their patterns of core, associated, and co-occurring traits, which we further validate and replicate in an independent cohort. We demonstrate that phenotypic and clinical outcomes correspond to genetic and molecular programs of common, de novo, and inherited variation, and further characterize distinct pathways disrupted by the sets of mutations in each class. Remarkably, we discover that class-specific differences in the developmental timing of impacted genes align with clinical outcome differences. These analyses embrace the phenotypic complexity of children with autism, unraveling genetic programs underlying their heterogeneity and suggesting specific biological dysregulation patterns and mechanistic hypotheses.

# **System Requirements** 

## Hardware requirements

The code in this repository requires only a standard computer with enough RAM to support the operations defined in the scripts.

## Software requirements

The conda environment used in the testing of all scripts is described in `conda_requirements.txt`, which contains all software dependencies, including version numbers. All code was tested on Linux operating systems. The latest version of the code has been tested on the following system: 

Linux: Rocky 8.10

We recommend starting a new conda environment using the requirements file:

```
conda create --name asd_env --file conda_requirements.txt
conda activate asd_env
conda list # verify the installation
```

Then, retrieve StepMix 1.2.5 from PyPI.

# Installation guide

We recommend cloning the repository to retrieve all scripts, executables, and supplementary data files (e.g. gene sets):

```
git clone https://github.com/FunctionLab/asd-pheno-classes.git
```

Cloning/installation time is less than 10s.

If you have access to SPARK and/or SSC data (see data availability statement below), you can execute the scripts by either:

(1) Replacing relative paths with local paths pointing to locations of phenotype and/or genotype datasets, or
(2) Copying phenotype datasets to the current directory to keep relative path references consistent.

Expected output: figures and tables will be in the figures/ subdirectory in each respective section. Some code may produce intermediate data files.

# Reproduction instruction

Given access to the data from SFARI Base, first resolve the paths using one of two strategies described above.

The scripts should be run in the following order to correctly reproduce the results:

  1) Execute `PreprocessingScripts/process_integrate_phenotype_data.py` to produce probands by phenotypes matrix. Make sure to correctly reference to SPARK phenotype dataset (`SPARK_collection_vX_date`).

      - Probands by phenotypes matrix can be found in `PhenotypeClasses/data/`.

  2) Execute `PhenotypeClasses/GFMM.py` to train and apply the model to the probands by phenotypes matrix, and obtain a label for each proband. Please allow some time for this script to run - we train 200 models with different initializations, but this should not take more than a couple of hours to run. This script produces:
  
      - A file with phenotypes and proband labels in `PhenotypeClasses/data/`.

          - ***Please note that in our analyses, class IDs correspond as follows: 0 -> Moderate challenges, 1 -> Broadly affected, 2 -> Social/behavioral, 3 -> Mixed ASD with DD. Class ID assignment is arbitrary and therefore, any trained model is likely to assign IDs in a different order.
    
      - A pie chart of class proportions (in `PhenotypeClasses/figures/`).
      
      - Age and sex breakdown by class (in `PhenotypeClasses/figures/`).
      
      - Horizontal lineplot summarizing the classes (in `PhenotypeClasses/figures/`).
      
      - Figure displaying variation of enrichment patterns in each class (in `PhenotypeClasses/figures/`).

  3) Execute `PhenotypeValidations/clinical_variable_validation.py`, which will produce:

      - Clinical validation plot (in `PhenotypeValidations/figures/`).
      
      - Parent-reported individual registration validation (in `PhenotypeValidations/figures/`).
      
      - SCQ and developmental milestones validation (in `PhenotypeValidations/figures/`).

  4) Given availability of SSC phenotype dataset, execute `PhenotypeValidations/clinical_variable_validation.py` to produce:

      - Replication figures (in `PhenotypeValidations/figures/`).

  5) DNV and inherited variant calling using HAT.

      - Execute `GenomicAnalyses.data_utils.get_WES_trios` to get valid trios (probands and siblings) for variant calling.
      
      - DNV calling outputs should be directed to `data/WES_V2_data/calling_denovos_data/output/`.

  6) Rare variant analyses can be executed as follows:

      - Run Ensembl's VEP on the variant calls from HAT.
      
      - Execute `GenomicAnalyses/variant_preprocessing_steps.py` to get all necessary data files for analysis.
      
      - Execute the rest of the scripts to reproduce figures from the paper:
        - `GenomicAnalyses/variant_set_enrichments.py`
        - `GenomicAnalyses/gene_constraint_analysis.py`
        - `GenomicAnalyses/gene_set_enrichments.py`
        - `GenomicAnalyses/odds_ratios.py`
        - `GenomicAnalyses/GO_term_analysis.py`
        - `GenomicAnalyses/developmental_trends_analysis.py`

## Software packages

We used the following software packages, which are publicly available for installation and use:

- HAT (downloaded from github on 10/11/22)
- plink v1.9
- ShinyGO 0.80
- StepMix 1.2.5 (retrieved from PyPI)
- Ensembl VEP tool (release 111.0)
- LOFTEE (v.1.0.4)
- AlphaMissense (VEP plugin)
- DeepVariant (v1.1.0)
- GATK HaplotypeCaller (v4.1.2.0)

### **Data Availability Statement**

In order to abide by the informed consents that individuals with autism and their family members signed when agreeing to participate in a SFARI cohort (SSC and SPARK), researchers must be approved by SFARI Base (https://base.sfari.org).

## Questions

Please direct any questions or requests to `aviya@princeton.edu` and `nsauerwald@flatironinstitute.org`.
