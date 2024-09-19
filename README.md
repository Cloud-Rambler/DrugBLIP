# Exploring drug mechanisms: a graph transformer-based model for protein-molecule interactions




## Requirements

Key dependencies include:

```bash
torch==2.0.1
transformers==4.35.0
deepspeed==0.12.2
pytorch-lightning==2.0.7
uni-core==0.0.1
```

See `requirements.txt` for more detailed requirements.

## Dataset

 Download the dataset, and put it under the `./data/` directory.

* **Stage1**
  * [stage1.zip](https://zenodo.org/records/13777651/files/stage1.zip?download=1)
  * [dude.zip](https://zenodo.org/records/13777651/files/dude.zip?download=1) for virtual screening
* **Stage2**
  * [stage2.zip](https://zenodo.org/records/13777651/files/stage2.zip?download=1)
* **Stage3**
  * [stage3.zip](https://zenodo.org/records/13777651/files/stage3.zip?download=1)

## Checkpoints

* **DrugBLIP**
  * [stage1.ckpt](https://zenodo.org/records/13777651/files/stage1.ckpt?download=1)
  * [stage1_ft.ckpt](https://zenodo.org/records/13777651/files/stage1_ft.ckpt?download=1)
  * [stage2_ft.ckpt](https://zenodo.org/records/13777651/files/stage2_ft.ckpt?download=1)
  * [stage3_ft.ckpt](https://zenodo.org/records/13777651/files/stage3_ft.ckpt?download=1)

* **Uni-Mol**
  * [mol_pre_no_h_220816.pt](https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt)
  * [pocket_pre_220816.pt](https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt)



## Reproduce the results

### Prerequisites

* Install required conda environment as described in **Requirements** section
* Download the dataset and required checkpoints as described in **Dataset** and **Checkpoints** section.

### Training the Model from Scratch

**Stage 1: 3D Pocket-Molecule Representation Learning**

Run the following script for stage 1 pretraining and fine-tuning:

```bash
bash ./scripts/stage1_pretrain_pair.sh
bash ./scripts/stage1_ft_pair.sh
```

**Stage 2: 3D Pocket-Molecule Alignment via Generative Learning**

Run the following script for stage 2 ft:

```bash
bash ./scripts/stage2_ft_pair.sh
```

**Stage 3: Docking Power Fine-tuning**

Run the following script for fine-tuning:

```bash
bash ./scripts/stage3_ft_pair.sh
```
**Training logs can be found in ./all_checkpoints**


### Evaluation on Our Pretrained Checkpoint

We share the checkpoint for reproducing results.

```bash
bash ./scripts/stage1_eval.sh
bash ./scripts/stage3_test.sh
```


## Citation

If you use our codes or checkpoints, please cite our paper:

```bib

```
