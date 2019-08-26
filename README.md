<p align="center"><img src="https://web.umons.ac.be/app/uploads/2018/02/UMONS-rouge-quadri-avec-texteth.png" width="480"\></p>

## PyTorch-GAN
Collection of PyTorch implementations of Generative Adversarial Network varieties presented in research papers. Model architectures will not always mirror the ones proposed in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GANs to implement are very welcomed.


## Table of Contents
  * [Installation](#installation)
  * [Execution sur CPU](#Execution sur CPU)
  * Paper [Conditional GAN](#conditional-gan)


## Installation (Procédure en français)
    1) Créer un compte sur Google Colaboratory à l'adresse https://colab.research.google.com/
	
	2) Importer le dernier notebook à partir du dépot https://github.com/rombaux/PyTorch-GAN
	
	3) Monter le Drive (ici, j'ai utilisé Gdrive
		from google.colab import drive
		drive.mount('/content/gdrive')
	
	4) Télécharger le Github
		!git clone https://github.com/rombaux/PyTorch-GAN
	
	5) Installer les dépendances
		cd /content/PyTorch-GAN
		!sudo pip3 install -r requirements.txt
	
	Vous pouvez maintenant choisir entre l'apprentissage et le test du modèle
	
	6) Apprendre un dataset
		cd /content/PyTorch-GAN/implementations/cgan/
		%run menu_train.py
		!python3 cgan_train.py --n_epochs $optn_epochs --sample_interval $optsample_interval --dataset $optdataset --channels $optchannel --n_classes $optn_classes --img_size $optimg_size --batch_size $optbatch_size
		%run printloss.py
		
	7) Tester le modèle	
		cd /content/PyTorch-GAN/implementations/cgan/
		%run menu_generate.py
		!python3 cgan_generate.py --dataset $optdataset --latent_dim 100 --channels $optchannels --genword $optgenword --img_size $optimg_size --n_classes $optn_classes
		%run printresult.py

	8) Suivez ensuite les instruction du menu.
		Pour l'apprentissage
		Choix du datataset
		Choix du batch size
		Choix de l'intervalle de générateur des samples
		Choix du nombre d'Epoch
		
		Pour le test
		Choix du datataset
		Choix du mot à générer

## Execution sur CPU (Procédure en français)
		
	cgan_generate_on_cpu.py --n_classes 10 --dataset 0 --channels 1 --img_size 32 --genword 648748454

### Conditional GAN
_Conditional Generative Adversarial Nets_

#### Authors
Rombaux Michael

#### Abstract
Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

[[Paper]](https://arxiv.org/abs/1411.1784) [[Code]](implementations/cgan/cgan.py)

