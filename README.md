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

 Download the dataset from [here](gg), and put it under the `./data/` directory.

## Checkpoints


Download following checkpoints from [here](gg), and put it under the `./all_checkpoints/` directory.

* **DrugBLIP**
  * stage1-ckpt
  * stage1-ft-ckpt
  * stage2-ft-ckpt
  * stage3-ft-ckpt

* **Uni-Mol**




## Reproduce the results

### Prerequisites

* Install required conda environment as described in **Requirements** section
* Download the dataset and required checkpoints as described in **Dataset** and **Checkpoints** section.



### Evaluation on Our Pretrained Checkpoint

We share the checkpoint for reproducing results.

```bash
bash ./scripts/stage3_test.sh
```

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

**Stage 3: Docking Fine-tuning**

Run the following script for fine-tuning:

```bash
bash ./scripts/stage3_ft_pair.sh
```

## Citation

If you use our codes or checkpoints, please cite our paper:

```bib

```
