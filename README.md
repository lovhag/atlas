# README

See [README_original.md](README_original.md) for the original Atlas README.

🎯 Goal: Evaluate Atlas base on ParaRel. 
    
A pre-trained Atlas model for cloze-style queries has not been released. However, it has been trained and evaluated on TempLAMA that is cloze-style, and we have the scripts for this. We can start to experiment with evaluating an Atlas model that has been fine-tuned on TempLAMA 2017 on ParaRel. Can validate our training setup by comparing with the reported results.

## Environment

```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate
pip install -r requriements.txt
pip install tensorboard==2.11.2
```

## Get the model

```bash
python preprocessing/download_model.py --model models/atlas/base --output_directory data
```

## Get the retrieval corpus

```bash
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2017 --output_directory data
```

## Get the precomputed retrieval indeces (maybe not needed)

```bash
python preprocessing/download_index.py --index indices/atlas/wiki/base --output_directory data
```

Specify with `--load_index_path`

## Get the tempLAMA dataset

```bash
python preprocessing/prepare_templama.py --output_directory data
```

## Time to create passage embeddings
 1st step 12 minutes, 256512 passages out of 26931153 (1%). Would take approx 21 hours to complete?
26931153/256512*12/60=21

For the full 2017 corpus, takes about 2 hours on 4 A40:4. Requires about 366GB in size.

For the 2018 corpus, requires only about 200GB?

### Background: the evaluation data (ParaRel)

Based on T-REx. 

TREx consists of 3.09 million Wikipedia abstracts aligned
with 11 million Wikidata triples. Wikidata from 2017-05 and Wikipedia abstracts from DBpedia (from 2016 at the latest).

Atlas authors observe that the 2018 December Wikipedia dump, 
which is close to the date of data collection for NQ, leads to the best results for both few-shot and full fine-tuning on NQ.

-> To best match the data time origin, should use the 2017 corpora for Atlas.

## Constrained decoding

Constrained decoding now implemented. Use `--use_decoder_choices` to enable this. See `alvis_eval_pararel.sh` for an example. Tested with [`alvis_test_choices.sh`](example_scripts/templama/alvis_test_choices.sh).

### Results on P449 evaluation (model finetuned on tempLAMA for 100 steps) 
Using pre-calculated index from BERT-base (not finetuned)? While retriever was finetuned to tempLAMA?

Without constrained decoding: 21.382 exact_match, see [results](data/experiments/858647-base-pararel-2017/P449-step-0.jsonl).

With constrained decoding: 75.814 exact_match. See [results](data/experiments/899796-base-pararel-2017/P449-step-0.jsonl).

Ran with [`alvis_eval_pararel.sh`](example_scripts/templama/alvis_eval_pararel.sh).

## Train Atlas on pararel

* 2017 corpora
* Use pre-encoded indeces for the base model with passages from 2017.
* [Script](alvis_scripts/train.sh) used.
* DECREASE EVAL FILE SIZE? (takes quite a lot of time)

Old approach: Use 'named-after' (P138) 9220 samples and 'original-network' (P449) 8811 samples. Amounts to a total of 18,031 samples. 

All training is performed without refreshing the index. Why we can reuse it also for evaluating the finetuned model.

## Evaluate Atlas on pararel

* Evaluate the model trained on pararel ([log](data/experiments/pararel-train-base-2017-900497)).
* [Script](alvis_scripts/eval.sh) used.

```bash
sbatch --array=0-27 alvis_scripts/eval.sh
```

After generating the evaluation files, the models can then be evaluated in the ParaRel project.

## Hyperparameter tuning for the few-shot training of Atlas

Tune batch size (32,64), learning rate ((5e-5,1e-5), (4e-5, 4e-5)), number of training steps (16, 32, 64, 128) and retriever temperature (0.1, 0.01).

Atlas authors decided on batch size 64, lr reader 5e-5, lr retriever 1e-5, retriever temp 0.1 and 64 train steps. For 770M model.

Specified by Atlas authors for TempLAMA, use: lr reader and retriever 4e-5, batch size 64, 100 training steps, retriever temperature 0.01  

Used by us for ParaRel tuning: same as above except for that the batch size is 4 (per GPU is 1).

Hyperparameter test span: 
* Batch size [4, 32, 64]
* Learning rate reader [5e-5, 4e-5]
    * Learning rate retriever [1e-5, 4e-5] (paired)
* Number of training steps [25, 50, 75, 100, 125] - will automatically do this for each script
* Retriever temperature [0.1, 0.01]
* Number of retrieved docs [10, 20, 30]

* Training data?
    * P138
    * (P449 (very few answer options))
    * (P138+P449)
    * P138+P127
    * P138+P127+P1412
Total of 108 possible settings (540 counting training steps). Do random sample over 22 settings (~20%) (110 counting training steps).

Evaluate on 100 randomly sampled entries from P17, P101, P264 and P449 each. (questions on if this gives the model an advantage?)

Decision: pick the run given by `data/experiments/pararel_training_hyperparam_search/12-987333` and `alvis_scripts/pararel_training_hyperparam_search/train_12.sh`. Performs generally good and does not require 3 training sets.

Evaluation of parameter- and fine-tuned model on full ParaRel done with `alvis_scripts/eval_tuned.sh`.

## Evaluate Atlas out-of-the-box on ParaRel

Atlas has been pretrained on the MLM task, so it should logically not need to be tuned to the MASK format of ParaRel. Thus, we should be able to evaluate Atlas out-of-the-box on ParaRel. This means that we can evaluate Atlas on all 30 ParaRel relations (we skip P37 which is not a N-1 relation).

However, the decoding of Atlas' predictions must be in the correct format for the model. Otherwise, it would be necessary to fine-tune the model to the format. The generations of Atlas look as follows `[0, 32099, 3, ...]`, so the constrained decoding (`prefix_allowed_tokens_fn`) must follow this format.

The current results for Atlas with retrieval can be found under `pararel-eval-zero-shot-base` and `hyperparam-tuned`. The current baseline results can be found under `pararel-eval-baseline-closed-book` and `hyperparam-tuned-closed-book`.

## Greedy branch

This is the greedy branch, denoting that we decode Atlas predictions in a greedy fashion. We here make use of the `prefix_allowed_tokens_fn` to constrain the answer generations by the model. Can e.g. observe for relation P138 that the model often predicts "arrow", "AIDS" or similarly, most likely due to that these answers start with "a" and that we have greedy decoding.

To run an evaluation, use `alvis_scripts/eval_no_space.sh`.
To debug the code, use `alvis_scripts/test_choices.sh`.