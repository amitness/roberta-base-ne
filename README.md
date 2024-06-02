# roberta-base-ne

Train `roberta-base` from scratch for Nepali language using the [CC-100](http://data.statmt.org/cc-100/) subset.

## Training
To start the training, run:
```shell
python train.py
```

The default configuration used is stored at `config/default.yaml`. You can also view all the configuration options using the `--help` command.
```shell
python train.py --help
```

You can override any configuration from the CLI using the [hydra](https://hydra.cc/docs/intro) syntax. For example, to train using only 100 sentences for 1 epoch, run:
```shell
python train.py dataset.portion=100 model.epochs=1
```

## Citations
Our model has been featured in the following papers:
- [Pande, Bishal Debb, et al. "Named Entity Recognition for Nepali Using BERT Based Models." International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems. Cham: Springer Nature Switzerland, 2023.](https://link.springer.com/chapter/10.1007/978-3-031-36822-6_8)
- [Niraula, Nobal, and Jeevan Chapagain. "DanfeNER-Named Entity Recognition in Nepali Tweets." The International FLAIRS Conference Proceedings. Vol. 36. 2023.](https://journals.flvc.org/FLAIRS/article/view/133384)
- [Timilsina, Sulav, Milan Gautam, and Binod Bhattarai. "NepBERTa: Nepali language model trained in a large corpus." Proceedings of the 2nd conference of the Asia-Pacific chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing. Association for Computational Linguistics (ACL), 2022.](https://aura.abdn.ac.uk/bitstream/handle/2164/21465/Timilsina_etal_ACLA_NepNERTa_VOR.pdf?sequence=1)
- [Tamrakar, Suyogya Ratna, and Chaklam Silpasuwanchai. "Comparative Evaluation of Transformer-Based Nepali Language Models." (2022).](https://www.researchsquare.com/article/rs-2289743/v1)
