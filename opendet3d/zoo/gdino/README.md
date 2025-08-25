# Grounding DINO

## Getting Started

- Test the pre-trained G-DINO model on COCO.
```bash
vis4d test --config opendet3d/zoo/gdino/gdino_swin_t_o365.py --gpus 1
```

- Train G-DINO with Objects365v1.

Please comment out [line 110](./gdino_swin_t_o365.py#L110) to train the model from scratch.
And refer to the MM Grounding DINO for data preprocessing.

```bash
vis4d fit --config opendet3d/zoo/gdino/gdino_swin_t_o365.py --gpus 8 --nodes 4 # batch size = 128
```

## Acknowledgement

This is the re-implementation of MM Grounding DINO please check the [original repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino) for more details.

## Citation

If you find this project useful in your research, please consider citing:

```latex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@article{zhao2024open,
  title={An Open and Comprehensive Pipeline for Unified Object Grounding and Detection},
  author={Zhao, Xiangyu and Chen, Yicheng and Xu, Shilin and Li, Xiangtai and Wang, Xinjiang and Li, Yining and Huang, Haian},
  journal={arXiv preprint arXiv:2401.02361},
  year={2024}
}