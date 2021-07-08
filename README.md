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