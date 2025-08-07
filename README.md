<p align="center">
  <a href="">
    <img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100" style="vertical-align: middle; margin-right: 50px;">
  </a>
  <a href="">
    <img src="https://gipsadoc.gricad-pages.univ-grenoble-alpes.fr/images/gipsa_logo.png" alt="GIPSA-logo" width="150" style="vertical-align: middle;">
  </a>
</p>

<h1 align='center'>Whistler Identification in Whistled Spanish (Silbo): A Case Study</h1>

<p align='center'>Alejandro López-García<sup>&dagger;</sup>, María Alfaro-Contreras<sup>&dagger;</sup>, Julien Meyer<sup>&ddagger;</sup>, Jose J. Valero-Mas<sup>&dagger;</sup>.</p>

&dagger; *Pattern Recognition and Artificial Intelligence Group, University of Alicante, Spain*

&ddagger; *Université Grenoble Alpes, CNRS, GIPSA-Lab, Grenoble, France*

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.11.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-white' alt='HuggingFace'>
</p>

<p align='center'>
  <a href='#about'>About</a> •
  <a href='#contents'>Contents</a> •
  <a href='#how-to-use'>How to use</a> •
  <a href='#citations'>Citations</a> •
  <a href='#acknowledgments'>Acknowledgments</a>
</p>

## About

This repository contains the source code and scripts to reproduce the experiments described in:

> **Whistler Identification in Whistled Spanish (Silbo): A Case Study"**  
> *Alejandro López-García, María Alfaro-Contreras, Julien Meyer, Jose J. Valero-Mas*  
> Proceedings of the 27th International Conference on Speech and Computer (SPECOM), 2025.

This work represents the first proposal for closed-set speaker identification in the context of whistled languages and, more precisely, the whistled Spanish of the Canary Islands (Silbo):

1. For the feature-based characterization of the samples, we consider the signal processing-based Mel-Frequency Cepstral Coefficients together with neural-based embedding methods based on Whisper and Wav2vec.

2. Different classifiers are assessed and compared for the task.

3. Several mechanisms for palliating data imbalance are studied.

## Contents

- *src/* : Source code.
- *features/* : Results of the feature extraction processes for each configuration considered.
- *run_experiments_\*.sh*: Shell scripts for executing the experiments.
- *requirements.txt*: Necessary libraries for the code.

## How to use

For the reproduction of the experiments included in the paper, please proceed as follows:
1. Install the requirements:
```
$ pip install -r requirements.txt
```

2. Execute the shell scripts that perform all experiments for each feature extraction process:
```
$ sh run_experiments_mfcc.sh
$ sh run_experiments_wav2vec.sh
$ sh run_experiments_whisper.sh
```

## Citations
Please, cite the work as:
<pre>
@inproceedings{LopezGarciaAlfaroContrerasMeyerValeroMas:SPECOM:2025,
  author = {López-García, Alejandro and Alfaro-Contreras, María and Meyer, Julien and Valero-Mas, Jose J.},
  title = {Whistler Identification in Whistled Spanish (Silbo): A Case Study},
  booktitle={Proceedings of the 27th International Conference on Speech and Computer},
  year = {2025},
}
</pre>


## Acknowledgments
This work was partially funded by the Generalitat Valenciana through project MUltimodal and Self-supervised Approaches for MUsic Transcription (CIGE/2023/216).