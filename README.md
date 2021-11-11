# MOLECULAR-TRANSLATION

<p align="center">
  <img src="./img/bristol.png">
</p>

This repository presents the solution developed during the **[BMS-Molecular-Translation competition](https://www.kaggle.com/c/bms-molecular-translation)**.
This one is composed of four parts:
* AutoEncoder
* Detector
* EncoderDecoder
* Initiator

These four parts were assembled in the final submission to provide an innovative and original solution 👍

---

## Contents

[***Objectives***](https://github.com/Fpiotro/MOLECULAR-TRANSLATION#objectives)

[***Evaluation***](https://github.com/Fpiotro/MOLECULAR-TRANSLATION#evaluation)

[***International Chemical Identifier Structure (InChI)***](https://github.com/Fpiotro/MOLECULAR-TRANSLATION#international-chemical-identifier-structure-inchi)

[***Model architecture***](https://github.com/Fpiotro/MOLECULAR-TRANSLATION#model-architecture)

[***Leaderboard***](https://github.com/Fpiotro/MOLECULAR-TRANSLATION#leaderboard)

[***Extras***](https://github.com/Fpiotro/MOLECULAR-TRANSLATION#extras)

## Objectives
* Automated recognition of optical chemical structures 
* Convert images back to the underlying chemical structure (InChI text) 
* Help chemists expand access to collective chemical research

## Evaluation

"[Levenshtein Distance](https://medium.com/@ethannam/understanding-the-levenshtein-distance-equation-for-beginners-c4285a5604f0) is defined as the minimum number of operations required to make the two inputs equal. Lower the number, the more similar are the two inputs that are being compared." ([Devopedia](https://devopedia.org/levenshtein-distance), 2021)

<p align="center">
  <img src="./img/evaluation.png">
</p>

## International Chemical Identifier Structure (InChI)

InChI is a non-proprietary, Open Source, chemical identifier intended to be an IUPAC approved and endorsed structure standard representation.

<p align="center">
  <img src="./img/Structure.png">
   <p align="center">
    Features of chemical structure in a hierarchical, layered manner. Major InChI layers: Main, Charge, Stereo, Isotopic, FixedH (never included in standard InChI) as well as the Reconnected layer (never included in standard InChI), and their associated sublayers.
  </p>
</p>

<p align="center">
  <b> This section was built with: <a href="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0068-4"> Heller, S.R., McNaught, A., Pletnev, I. et al. <i>InChI, the IUPAC International Chemical Identifier.</i> J Cheminform 7, 23 (2015). https://doi.org/10.1186/s13321-015-0068-4 </a>
  </b>
</p>

## Model architecture

<p align="center">
  <img src="./img/Architecture_0.png">
   <p align="center">
    Architecture of the model with the outputs associated to each branch. The final prediction is a combination of the three branches.
  </p>
</p>

<p align="center">
  <img src="./img/Transfer_Learning.png">
   <p align="center">
    Illustration of the Transfer Learning part between the Initiator branch and the EncoderDecoder branch. The Initiator branch is voluntarily trained on a simplified problem in order to accustom the Resnet101 with images of molecules. Therefore, the Initiator branch is trained before the EncoderDecoder branch in order to extract the weights from the Resnet101 Backbone and load them on the Resnet101 Encoder of the EncoderDecoder branch.
  </p>
</p>

## Leaderboard

<p align="center">
  <img src="./img/Score.png" width=50% height=50%>
</p>

## Extras

### 2D and 3D representation of molecules

Representing the molecule in three dimensions allows to understand the structure of the molecule, interactions between bonds and atoms. See [`Extras.ipynb`](https://github.com/Fpiotro/MOLECULAR-TRANSLATION/blob/main/Extras.ipynb) (💥Notebook made with the Kaggle platform, package installations are different if you use Google Colab💥) to be able to plot molecules in 3D.

2D Structure             |  3D Structure
:-------------------------:|:-------------------------:
![](./img/RDKIT_SMILES.png)  |  ![](./img/Mol_3D_1.png)
![](./img/RDKIT_SMILES_2.png)  |  ![](./img/Mol_3D_2.png)
![](./img/RDKIT_SMILES_3.png)  |  ![](./img/Mol_3D_3.png)

---
