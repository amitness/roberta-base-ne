# roberta-base-ne

Train `roberta-base` from scratch for Nepali language using the [CC-100](http://data.statmt.org/cc-100/) subset.

## Training
To start the training with default settings, run:
```shell
python train.py
```

To view all available configuration options, use the `--help` command:
```shell
python train.py --help
```

You can override any default configuration from the command line. For example, to train for 1 epoch using 50% of the data, run:
```shell
python train.py --dataset-portion "50%" --epochs 1
```

All default values are defined in the script's argparse configuration.

## Citations

Our model has been featured in the following papers and work:

[1] Poudel, S., Rauniyar, K., Acharya, A., Rashid, J., Adhikari, S., Naseem, U., & Thapa, S. (2025). [NepAES: Exploring the promise of automated essay scoring for Nepali essays](https://peerj.com/articles/cs-3253/). *PeerJ Computer Science*, 11, e3253.

[2] Khadka, P., Bk, A., Acharya, A., K.c., B., Shrestha, S., & Thapa, R. (2025). [Nepali Transformers@NLU of Devanagari Script Languages 2025: Detection of Language, Hate Speech and Targets](https://aclanthology.org/2025.chipsal-1.36/). In *Proceedings of the First Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2025)* (pp. 314-319). Abu Dhabi, UAE.

[3] Paudel, A., Puri, A., & Sigdel, Y. (2024). [TrOCR Devanagari - Handwritten Text Recognition](https://huggingface.co/paudelanil/trocr-devanagari-2). *Hugging Face Model Card*.

[4] Pande, B. D., Hettiarachchi, H., & Premasiri, D. (2023). [Named Entity Recognition for Nepali Using BERT Based Models](https://link.springer.com/chapter/10.1007/978-3-031-36822-6_8). In *International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems* (pp. 123-132). Springer Nature Switzerland.

[5] Niraula, N., & Chapagain, J. (2023). [DanfeNER - Named Entity Recognition in Nepali Tweets](https://journals.flvc.org/FLAIRS/article/view/133384). *The International FLAIRS Conference Proceedings*, 36(1).

[6] Timilsina, S., Gautam, M., & Bhattarai, B. (2022). [NepBERTa: Nepali language model trained in a large corpus](https://aura.abdn.ac.uk/bitstream/handle/2164/21465/Timilsina_etal_ACLA_NepNERTa_VOR.pdf?sequence=1). In *Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing* (pp. 1463-1472). Association for Computational Linguistics.

[7] Tamrakar, S. R., & Silpasuwanchai, C. (2022). [Comparative Evaluation of Transformer-Based Nepali Language Models](https://www.researchsquare.com/article/rs-2289743/v1). *Research Square Preprint*.
