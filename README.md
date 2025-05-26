# FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL'24-main, Long Paper)

Here is our paper on arXiv: [[link](https://arxiv.org/abs/2407.00908)]

The structure of the projects:
- dataset: FRANK and REALSumm (in JSON format) are located in this folder
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2 
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models

## Highlight

**FineSurE** is a *multi-dimensional*, *fine-grained* automated evaluation framework for text summarization. It covers there distinctive evaluation dimensions, namely faithfulness, completeness, and conciseness. These dimensions are crucial to assess the capability of modern language models in summarization, as they are susceptible to incorrect statement, information omission, and verbosity.

FineSurE framework breaks down a complicate evaluation process into ```two simple human-like evaluation tasks``` using LLMs. 

- **Fact Checking:** This is a task of solving a categorization problem involving nine categories. These include the seven factuality errors, along with an additional category "other error" for error outside the seven errors, and an additional category "no error" for cases whether no error was detected. Given a pair of input text and model summary, the LLM is expected to output the error type classified into one of the nine categories for each sentence along with a concise reason.
  
- **Keyfact Alignment:** This is an alignment task of matching each keyfact into the summary sentences from which the keyfact is inferable. Given a pari of keyfact list and model summary, the output should be the binary label (whether inferable or not) an dth elist of line numbers of all summary sentences matched for each keyfact.

<p align="center">
<img width="755" alt="스크린샷 2024-07-02 오후 4 56 27" src="https://github.com/DISL-Lab/FineSurE-ACL24/assets/10972556/e5c733b7-d863-4e39-92ac-98f63e8bbee5">
</p>

## Running FineSurE on Model Summareis

We create sample datasets with 10 examples for fact-checking and keyfact-alignment tasks, respectively.

Please replace the ```openai api key``` with your api key in ```finesure/fact-checking.py``` and ```finesure/keyfact-alignmnet.py```.

#### Runnining Command:
```bash
cd CodeRelease
python finesure/fact-checking.py [input-path] [output-folder]

# example code for fact checking on sampled data.
python finesure/fact-checking.py dataset/frank/frank-data-sample-10.json result/fact-checking
```

#### Runnining Command:
```bash
cd CodeRelease
python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]

# example code for keyfact alignment on sampled data.
python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data-sample-10.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
```

#### Logs:

The results are saved in the result directory. See the results on examples below:

* Fact Checking Task:
```bash 
[Evaluation Results]
* sentence-level factuality error ratio per model (lower is better)
bert_sum	0.0%
bus	33.3%
pgn	16.7%
s2s	83.3%
bart	33.3%

* summary-level faithfulness score per model (higher is better)
bert_sum	100.0%
bus	66.7%
pgn	83.3%
s2s	16.7%
bart	75.0%

* system-level model ranking (left is better)
['bert_sum', 'pgn', 'bart', 'bus', 's2s']

* success rate: 100.0%
```

* Keyfact Alignment Task:
```bash 
[Evaluation Results]

* completeness score per model (higher is better)
unilm_out_v2	45.5%
t5_out_large	59.0%

* completeness model ranking (left is better)
['t5_out_large', 'unilm_out_v2']

* conciseness score per model (higher is better)
unilm_out_v2	76.0%
t5_out_large	81.7%

* conciseness model ranking (left is better)
['t5_out_large', 'unilm_out_v2']

* success rate: 100.0%
```

## Reproduce the Main Table of the Paper

```bash
cd CodeRelease/reproduce
python reproduce-main-results.py results/frank-result-by-gpt4-w-finesure.json results/realsumm-result-by-gpt4-w-finesure.json
```


## Citation

Please consider citation if our paper is useful in your research.

```BibTeX
@inproceedings{song2024finesure,
  title={FineSurE: Fine-grained Summarization Evaluation using LLMs},
  author={Song, Hwanjun and Su, Hang and Shalyminov, Igor and Cai, Jason and Mansour, Saab},
  booktitle={ACL},
  year={2024}
}
```


| 모델 (KeyFact 출처)                | Faithfulness bAcc | Faithfulness (P / S) | Faithfulness RankCorr | Completeness (P / S) | Completeness RankCorr | Conciseness (P / S) | Conciseness RankCorr | Success Ratio |
| ------------------------------ | ----------------- | -------------------- | --------------------- | -------------------- | --------------------- | ------------------- | -------------------- | ------------- |
| GPT-3.5 Turbo (Human)          | 67.1%             | 0.668 / 0.720        | 0.898                 | 0.597 / 0.519        | 0.678                 | 0.381 / 0.366       | 0.538                | 96.6%         |
| GPT-3.5 Turbo (Machine)        | 67.1%             | 0.668 / 0.720        | 0.898                 | -0.409 / -0.333      | N/A                   | 0.134 / 0.214       | N/A                  | 8.2%          |
| GPT-4 1106 (Human)             | 88.4%             | 0.874 / 0.890        | 0.902                 | 0.710 / 0.690        | 0.798                 | 0.399 / 0.338       | 0.424                | 100.0%        |
| GPT-4 1106 (Machine)           | 88.4%             | 0.874 / 0.890        | 0.902                 | 0.682 / 0.735        | N/A                   | 0.729 / 0.691       | N/A                  | 6.0%          |
| GPT-4.1 (Human)                | 88.2%             | 0.845 / 0.847        | 0.945                 | 0.724 / 0.700        | 0.852                 | 0.559 / 0.534       | 0.614                | 100.0%        |
| GPT-4.1 (Machine)              | 88.2%             | 0.845 / 0.847        | 0.945                 | 0.101 / 0.273        | N/A                   | 0.080 / 0.087       | N/A                  | 11.0%         |
| GPT-4.1 Mini (Human)           | 89.7%             | 0.903 / 0.913        | 0.987                 | 0.748 / 0.735        | 0.785                 | 0.554 / 0.510       | 0.768                | 100.0%        |
| GPT-4.1 Mini (Machine)         | 89.7%             | 0.903 / 0.913        | 0.987                 | -0.215 / -0.265      | N/A                   | -0.131 / -0.118     | N/A                  | 8.5%          |
| GPT-4.1 Nano (Human)           | 83.5%             | 0.837 / 0.842        | 0.904                 | 0.662 / 0.657        | 0.570                 | 0.383 / 0.310       | 0.157                | 98.5%         |
| GPT-4.1 Nano (Machine)         | 83.5%             | 0.837 / 0.842        | 0.904                 | 0.256 / 0.339        | N/A                   | 0.615 / 0.579       | N/A                  | 5.0%          |
| GPT-4o-24-05-13 (Human)        | 89.8%             | 0.864 / 0.868        | 0.967                 | 0.692 / 0.868        | 0.824                 | 0.512 / 0.454       | 0.780                | 100.0%        |
| GPT-4o-24-05-13 (Machine)      | 89.8%             | 0.864 / 0.868        | 0.967                 | 0.076 / 0.095        | N/A                   | -0.184 / -0.242     | N/A                  | 9.5%          |
| GPT-o3-25-04-16 (Human)        | 89.1%             | 0.852 / 0.845        | 0.870                 | 0.785 / 0.767        | 0.737                 | 0.507 / 0.487       | 0.332                | 100.0%        |
| GPT-o3-25-04-16 (Machine)      | 89.1%             | 0.852 / 0.845        | 0.870                 | -0.143 / -0.261      | N/A                   | -0.181 / -0.236     | N/A                  | 8.5%          |
| GPT-o4-mini-25-04-16 (Human)   | 88.0%             | 0.863 / 0.872        | 0.920                 | 0.666 / 0.663        | 0.740                 | 0.439 / 0.422       | 0.568                | 100.0%        |
| GPT-o4-mini-25-04-16 (Machine) | 88.0%             | 0.863 / 0.872        | 0.920                 | -0.261 / -0.353      | N/A                   | 0.305 / 0.334       | N/A                  | 6.0%          |
