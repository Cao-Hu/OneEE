# OneEE: A One-Stage Framework for Fast Overlapping and Nested Event Extraction

Source code for COLING 2022 paper: Unified Named Entity Recognition as Word-Word Relation Classification

> Event extraction (EE) is an essential task of information extraction, which aims to extract structured event information from unstructured text. Most prior work focuses on extracting flat events while neglecting overlapped or nested ones. A few models for overlapped and nested EE includes several successive stages to extract event triggers and arguments, which suffer from error propagation. Therefore, we design a simple yet effective tagging scheme and model to formulate EE as word-word relation recognition, called OneEE. The relations between trigger or argument words are simultaneously recognized in one stage with parallel grid tagging, thus yielding a very fast event extraction speed. The model is equipped with an adaptive event fusion module to generate event-aware representations and a distance-aware predictor to integrate relative distance information for word-word relation recognition, which are empirically demonstrated to be effective mechanisms. Experiments on 3 overlapped and nested EE benchmarks, namely FewFC, Genia11, and Genia13, show that OneEE achieves the state-of-the-art (SOTA) results. Moreover, the inference speed of OneEE is faster than those of baselines in the same condition, and can be further substantially improved since it supports parallel inference.

## 1. Environments

```
- python (3.8.12)
- cuda (11.4)
```

## 2. Dependencies

```
- numpy (1.19.2)
- torch (1.10.0)
- transformers (4.10.0)
- prettytable (2.1.0)
```

## 3. Dataset

- FewFC: Chinese Financial Event Extraction dataset. The original dataset can be accessed at [this repo](https://github.com/TimeBurningFish/FewFC). Here we follow the settings of [CasEE](https://github.com/JiaweiSheng/CasEE). Note that the data is avaliable at /data/fewFC, and we adjust data format for simplicity of data loader. To run the code on other dataset, you could also adjust the data as the data format presented.
- [ge11: GENIA Event Extraction (GENIA), 2011](https://2011.bionlp-st.org/home/genia-event-extraction-genia)
- [ge13: GENIA Event Extraction (GENIA), 2013](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki/Overview)

## 4. Preparation

- Download dataset
- Process them to fit the same format as the example in `data/`
- Put the processed data into the directory `data/`

## 5. Training

```bash
>> python main.py --config ./config/fewfc.json
```
## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 7. Citation

If you use this work or code, please kindly cite this paper:

```
@inproceedings{cao2022oneee,
  title={OneEE: A One-Stage Framework for Fast Overlapping and Nested Event Extraction},
  author={Cao, Hu and Li, Jingye and Su, Fangfang and Li, Fei and Fei, Hao and Wu, Shengqiong and Li, Bobo and Zhao, Liang and Ji, Donghong},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={1953--1964},
  year={2022}
}
```
