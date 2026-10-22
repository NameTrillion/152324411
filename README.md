
### Local Play (Command Line)

Download or `git clone` this repository. From the repository root:

```
pip install -e ".[jax,tf]"

python scripts/eval_two.py --p1.type human --p2.ai.path <path/to/trained/model> [--dolphin.copy_home_directory]

python scripts/eval_two.py --help  # to get a full list of options
```

You can change the character by setting `--p2.character <fox/falco/marth/...>`. The full set of released models is in this [Google Drive folder](https://drive.google.com/drive/folders/1etYN_IgVoUPleAsh76s_9443e4XzGKWo?usp=sharing).

#### Notes
* Tested with python 3.12-3.14.
* By default phillip sets up human players as using Wii-U controller adapters. If you want to use your own dolphin configuration (including controller config) pass `--dolphin.copy_home_directory`. You still need to specify which player (p1 or p2) is human.
* On Windows you may need to [enable long paths](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#registry-setting-to-enable-long-paths) in order for the pip installs to work.
* On Windows make sure no other dolphin instances are running as it will prevent the bot from sending inputs to the game.
* The script will try to find slippi dolphin and your melee ISO; if it fails you can manually specify `--dolphin.iso` and `--dolphin.path`.


# Code Overview

Phillip is trained in two stages. In the first stage, it learns to imitate human play from a large dataset of slippi replays. The resulting imitation policy is ok, but makes a lot of mistakes. In the second stage, the imitation policy is refined by playing against itself with Reinforcement Learning. This results in much stronger agents that have their own style of play.


## Imitation Learning

The entry point for imitation learning is [`scripts/train.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/train.py). See [`scripts/imitation_example.sh`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/imitation_example.sh) for appropriate arguments.

Metrics are logged to [wandb](https://wandb.ai/) during training. To use your own wandb account, set the `WANDB_API_KEY` environment variable. The key metric to look at is `eval.policy.loss` -- once this has plateaued you can stop training. On a good GPU (e.g. a 3080Ti), imitation learning should take a few days to a week. The agent checkpoint will be periodically written to `experiments/<tag>/latest.pkl`.

## Evaluation

To play a trained agent or watch two trained agents play each other, use [`scripts/eval_two.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/eval_two.py). To do a full evaluation of two agents against each other, use [`scripts/run_evaluator.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/run_evaluator.py).
