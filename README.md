# SAM3-Adapter-for-CDW-waste-seg

This repo contains the supported pytorch code and configurations for the "Adapter-based Fine-tuning of SAM3 for Construction and Demolition Waste Segmentation on Conveyor Belts" article. 

## Abstract
Construction and demolition (C&D) waste sorting on conveyor belts is challenging because materials are irregularly shaped, visually similar, overlapping, and presented in cluttered scenes. To address these challenges, we develop a parameter-efficient adaptation of the Segment Anything Model 3 (SAM3) for multiclass semantic segmentation of C&D waste, supporting automated material recognition for conveyor-based sorting systems. We evaluate the approach on the ReCoDeWaste dataset, which contains more than 110,000 annotated masks across six C&D waste material classes. We first evaluate zero-shot SAM3 as a reference baseline. We then incorporate lightweight adapters into the frozen SAM3 image encoder and jointly optimise the adapters, mask decoder, and multiclass prediction head for class-specific pixel prediction. Component-wise experiments isolate the contribution of the adapter modules. Our full SAM3-Adapter framework achieves an mIoU of 0.677, F1-score of 0.803, and overall accuracy of 0.825, compared with 0.518, 0.744, and 0.756, respectively, for the strongest fine-tuned configuration without adapters. We optimise approximately 4.3 million of the 458 million parameters (0.94%).  These results demonstrate that parameter-efficient domain adaptation can substantially improve multiclass waste segmentation while retaining a frozen foundation-model backbone, providing a perceptual foundation for future robotic waste sorting systems.


## System requirements
This code was implemented with Python 3.12.11 and PyTorch 2.7.1. You can install all the requirements via:

pip install -r requirements.txt

## Quick strat

1.	Download the dataset and split it into training, validation and testing.
2.	Download the pre-trained SAM3 and put it in ./pretrained folder.
3.	Training:

```bash
torchrun --nproc_per_node=1 Train.py --config [CONFIG_PATH]
```

5.	Evaluation:

```bash
torchrun --nproc_per_node=1 Test.py --config [CONFIG_PATH] --model [FINE-TUNED_MODEL_PATH]
```

7.	Download trained model weights from this shared (LINK)


##Dataset 
ReCoDe dataset (https://github.com/prasadvineetv/ReCoDeWaste-Dataset)

## Acknowledgement

This work was inspired by and builds on the following research works:
[SAM 3: Segment Anything with Concepts]([https://github.com/facebookresearch/sam3](https://arxiv.org/abs/2511.16719))
[SAM3-Adapter: Efficient Adaptation of Segment Anything 3 for Camouflage Object Segmentation, Shadow Detection, and Medical Image Segmentation](https://arxiv.org/abs/2511.19425)
[A benchmark dataset for classwise segmentation of construction and demolition waste in cluttered environments](https://doi.org/10.1038/s41597-025-05243-x)


