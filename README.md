# Evaluating Atlas for factual consistency

See [README_original.md](README_original.md) for the original Atlas README.

This repo describes how we evaluate Atlas-base and Atlas-large on ParaRel to measure the factual consistency of the retrieval-augmented models.

## Setup

Before it is possible to run the scripts described further down, you need to setup the environment, download the models and retrieval corpus and pre-calculate the indices for the retrieval corpus.

### Environment

Create a virtual environment `venv` in the directory folder and run the following commands: 
```bash
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
source venv/bin/activate
pip install -r requriements.txt
pip install tensorboard==2.11.2
```

We assume here that you are using a HPC with a pre-installed module that contains Pytorch. If not, you should skip the `module load` line and install Pytorch separately.

### Download the models and retrieval corpus

Create a folder named `data` in the working directory. Then run the following commands using the tools provided in the original Atlas repo to download the Atlas models: 
```bash
python preprocessing/download_model.py --model models/atlas/base --output_directory data
python preprocessing/download_model.py --model models/atlas/large --output_directory data
```

We will use the Wikipedia 2017 dump as the retrieval corpus for Atlas. Download it using the following command:
```bash
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2017 --output_directory data
```

> Atlas authors observe that the 2018 December Wikipedia dump, which is close to the date of data collection for NQ, leads to the best results for both few-shot and full fine-tuning on NQ. Since we will evaluate Atlas on ParaRel, we use the 2017 December Wikipedia dump instead. ParaRel is based on T-REx that consists of 3.09 million Wikipedia abstracts aligned with 11 million Wikidata triples from 2017-05 Wikidata and Wikipedia abstracts from DBpedia (from 2016 at the latest), and should therefore better align with a knowledge source dated around 2017.

### Pre-calculate the indices for Wikipedia 2017

We need to run the Atlas models several times in pararell on different evaluation data. To make this more efficient, we pre-compute the retrieval indices. Since we do not train the Atlas reader or retriever, we can use the same pre-computed indices throughout.

Use [alvis_scripts/processing/get_base_passage_embeddings.sh](alvis_scripts/processing/get_base_passage_embeddings.sh) and [alvis_scripts/processing/get_large_passage_embeddings.sh](alvis_scripts/processing/get_large_passage_embeddings.sh) to compute and save the indices for Atlas-base and -large respectively. The embeddings will be saved to [data/saved_index/](data/saved_index/).

Time to create and store the indices for base is 2 hours on 4 A40:4 and requires a storage space of 366 GB. Time to create and store the indices for large is 47 minutes on 8 A40:4 and requires a storage space of 192 GB. 

## ParaRel

This project lets us evaluate Atlas on ParaRel. 

First, we generate and format the ParaRel data and query files in our separate [ParaRel project](https://github.com/dsaynova/pararel/tree/main). Then we evaluate our Atlas models on the queries, performing one run per relation to evaluate for.

After generating the ParaRel prediction files per relation we get our evaluation results from the [ParaRel project](https://github.com/dsaynova/pararel/tree/main).

### Create small data sets for testing

Running the full Atlas model with full-sized evaluation data and retrieval corpus takes a significant amount of time, why we also make use of smaller test sets for checking that the code runs as expected. Apart from the original evaluation data, we also generate and make use of the following data for testing:

- Debug passages: apart from the standard `enwiki-dec2017` corpus, we create a smaller version of it `enwiki-dec2017-DEBUG` by extracting the topmost 100 lines of the `infobox.jsonl` and `text-list-100-sec.jsonl` and putting them in files with the same names in the debug folder.
- Debug queries: we create smaller debug versions of ParaRel queries for different relations. For example, we extract the 100 lines of file `P140.jsonl` and save it to `P140_100.jsonl` with its corresponding options file `P140_100_options.txt`.

## Decoding Atlas predictions to ParaRel queries

ParaRel was initially developed as an cloze-style task for encoder models. We wish to evaluate Atlas zero-shot with a restricted candidate set on this. Atlas is an encoder-decoder model that has been trained with an MLM pretext task and to some extent been evaluated zeros-shot in an MLM setup on MMLU, but this does not perfectly match the ParaRel format. We therefore need to make some adaptations of ParaRel and Atlas:

1. Reformat all ParaRel queries from e.g. "Kings Domain, located in [MASK]." to "Kings Domain, located in<extra_id_0>." by replacing the [MASK] with the correct mask token expected by Atlas and removing preceding spaces.
2. At Atlas inference time: calculate the likelihood estimations for each entry in the restricted candidate set and pick the candidate that maximizes the likelihood. See the functions `generate_from_choices_by_likelihood` and `get_choice_generator_values` in [src/atlas.py](src/atlas.py) that largely implement this for more details. 
    - Use `--use_decoder_choices` to enable this. The program will look for a file named "<data_path>_options.txt" to get the candidate set. A small test of the functionality can be performed using [alvis_scripts/test_choices.sh](alvis_scripts/test_choices.sh).
    - This decoding strategy results in a code that is slower to run, as we need to evaluate all answer candidates. 
    - This decoding strategy also means that the setup is more sensitive to using the correct format for the answer alternatives. 
        - Some extra formatting we perform is to remove the [EOS] token from the maximum likelihood estimation, as the Atlas model is prone to generate approximately three MLM predictions per query and might be confused when only prompted to generate one. We also add answer options for which an extra padding "3" token has been added, preceding each answer candidate.
        - We verify that the process works as intended by comparing the constrained Atlas predictions to free-form generations to the same queries. If the free-form generation happens to match any of the allowed candidates and does not match the constrained decoding, something is off with the decoding. Our current decoding approach resulted in a minimum number of such mismatches.
    - This also means that we avoid the issues caused by e.g. a greedy decoding, in that the model might be more prone to generate certain answers just because they start with an "a".

## Evaluation on ParaRel

See the evalluation method for each model below.

### Atlas-base

Evaluation of Atlas-base zero-shot on ParaRel is done using the following command:

```bash
sbatch --array=0-29 alvis_scripts/eval_no_space.sh
```

For testing, use `alvis_scripts/test_error_cases.sh`.

### Atlas-large

This model has 770M parameters (compared to the 220M parameters of Atlas-base). Use the following code to evaluate it.

```bash
sbatch --array=0-29 alvis_scripts/eval_large.sh
```

### Baselines

We also evaluate Atlas baselines on Pararel. Use the following code to generate model baseline predictions:
* Atlas closed-book: `baselines/eval_closed_book.sh`
* T5-base: `baselines/eval_baseline_t5.sh`

### Results
The results per relation is saved in a folder for each model. The ParaRel predictions are saved to e.g. "P17-step-0.jsonl" for relation P17.

- Atlas-base: `data/experiments/pararel-eval-zero-shot-no-space-likelihood-no-eos-with-3`
- Atlas-large: `data/experiments/pararel-eval-zero-shot-large`
- Atlas-base closed-book: `data/experiments/pararel-eval-zero-shot-base-no-space-likelihood-no-eos-with-3-closed-book`
- T5 as baseline: `data/experiments/pararel-eval-baseline-t5-no-space-likelihood-no-eos-with-3`


## Evaluation on ParaRel with fixed passages

We also evaluate Atlas-base on ParaRel for which we have fixed the passages to retrieve, overruling the Atlas retriever. To run Atlas with fixed retrieved passages use the `--use_file_passages` option. The code then assumes that a field `passages` can be found in the evaluation data. The evaluation data with fixed passages is generated in the separate [ParaRel project](https://github.com/dsaynova/pararel/tree/main).

We experiment with three different settings for fixing the retrieved passages. 
1. We take the retrieved passages for one of the queries for a certain fact triplet and and use those for all queries for that fact (`alvis_scripts/eval_fixed_retrieval.sh`), making the retrieval both relevant and consistent. 
2. We take the retrieved passages for another query for the same relation as the current query (`alvis_scripts/eval_fixed_retrieval_semi_random.sh`), making the retrieval consistent and cohesive across the retrieved passages but irrelevant. 
3. We use a set of completely random passages as the retrieval (`alvis_scripts/eval_fixed_retrieval_random.sh`), making it consistent but irrelevant and incohesive.

## Compute retriever embeddings for retriever consistency measurements

In the [ParaRel project](https://github.com/dsaynova/pararel/tree/main) we also investigate to what extent the retriever is consistent. To enable retriever consistency metrics with respect to embedding similarities between instances of retrieved passages, we need to compute the retriever embeddings for the ParaRel queries.

To compute and store the embeddings, run the following code. 

```bash
sbatch --array=0-29 alvis_scripts/compute_retriever_embeddings/atlas_base.sh
```

and 

```bash
sbatch --array=0-29 alvis_scripts/compute_retriever_embeddings/atlas_large.sh
```

The embeddings will then be saved to the same folder as the ParaRel predictions. The code can be tested using [alvis_scripts/compute_retriever_embeddings/test_atlas_base.sh](alvis_scripts/compute_retriever_embeddings/test_atlas_base.sh).

## Train Atlas on pararel (not used in submitted paper)

> This approach was not used in the paper submitted as it gave Atlas an advantage on ParaRel compared to other models, making it hard to compare its results to those of other models evaluated on ParaRel. Also, this approach was used prior to the adaption of the maximum likelihood based decoding, before the query format had been successfullly updated to Atlas. The approach used for the submitted paper relied on evaluating Atlas zero-shot.

We experimented with fine-tuning Atlas on two ParaRel relations, 'named-after' (P138) with 9220 samples and 'original-network' (P449) with 8811 samples. [alvis_scripts/train.sh](alvis_scripts/train.sh) was used for this. All training is performed without refreshing the index. Why we can reuse it also for evaluating the finetuned model.

We also experimented with tuning the hyperparameters for the Atlas fine-tuning on ParaRel. The Atlas authors decided on lr reader and retriever 4e-5, batch size 64, 100 training steps and retriever temperature 0.01 for the tuning on TempLAMA. We investigated the following hyperparameters and value ranges:
* Batch size [4, 32, 64]
* Learning rate reader [5e-5, 4e-5]
    * Learning rate retriever [1e-5, 4e-5] (paired)
* Number of training steps [25, 50, 75, 100, 125, 150] (will automatically do this for each script)
* Retriever temperature [0.1, 0.01]
* Number of retrieved docs [10, 20, 30]

We also investigated to what extent the choice of fine-tuning data impacted the validation performance of the model and experimented with tuning over the following splits:
* P138
* P138+P127
* P138+P127+P1412

This amounts to a total of 108 possible hyperparameter settings (540 counting training steps). We perform a random sample over 22 settings (~20%) (110 counting training steps).

The tuned models are then evaluated on 100 randomly sampled entries from P17, P101, P264 and P449 each. 

### Results

The tuning scripts can be found under [alvis_scripts/pararel_training_hyperparam_search](alvis_scripts/pararel_training_hyperparam_search). By examination of the validation results in tensorboard we made the following general findings:
* Maximum performances (out of 100) for the different validation sets are P17: 98, P101: 64, P264: 75 and P449: 90.
* More train steps give better results, between 100 and 150 steps is preferable while this also depends on the batch size used.
* Batch size has no clear effect.
* A higher reader lr is better.
* More retrieved passages improves performance on P264 and P449.
* Higher temperatures are better.

The best performing run was run 12 with 100 train steps, reader lr 5e-5, retriever lr 1e-5, batch size 32, P138 and P127 train data, 0.1 retriever temperature and 30 retrieved passages. Validation performance on P17: 97, P101: 59, P264: 75 and P449: 91.

Evaluation of parameter- and fine-tuned model on full ParaRel done with `alvis_scripts/eval_tuned.sh`.