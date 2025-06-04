# _FineSurE_ Reproduction
This repository aims to re-implement and enhance upon the [FineSurE: Fine-grained Summarization Evaluation using LLMs (ACL '24-main, Long Paper)](https://arxiv.org/abs/2407.00908)

## 🚀 Quick Start Guide
0. **Clone this Repo**
    ```bash
    git clone https://github.com/wjkim9653/FineSurE-ACL24.git
    ```
1. **Create and save .env file with following contents (write your own OPENAI API Key)**
    ```bash
    OPENAI_API_KEY=''
    ```
2. **To Quickly Test-Run FineSurE Framework as a whole and reproduce the Table 1, 2 from the paper, simply run:**
    ```bash
    ./run_all.sh
    ```
    - By default, it runs just 50 samples (randomly sampled) to reduce API Cost.
    - To modify the number of samples, change the `SAMPLE_COUNT` variable within `run_all.sh`
    - Real-Time Logs from running FineSurE's Faithfulness/Completeness/Conciseness Evaluations (for each LLM Judges) can be found under the directory: **`result/`**
3. **After FineSurE framework runs, the results are stored in `/reproduce/logs`**

    The table below is the result from running `./run_all.sh` with `SAMPLE_COUNT=200`

    | 모델 (KeyFact 출처)                | Faithfulness bAcc | Faithfulness (P / S) | Faithfulness RankCorr | Completeness (P / S) | Completeness RankCorr | Conciseness (P / S) | Conciseness RankCorr | Success Ratio |
    | ------------------------------ | ----------------- | -------------------- | --------------------- | -------------------- | --------------------- | ------------------- | -------------------- | ------------- |
    | GPT-3.5 Turbo (Human)          | 67.1%             | 0.668 / 0.720        | 0.898                 | 0.613 / 0.539        | 0.522                 | 0.386 / 0.370       | 0.369                | 100.0%        |
    | GPT-4 1106 (Human)             | 88.4%             | 0.874 / 0.890        | 0.902                 | 0.710 / 0.690        | 0.795                 | 0.399 / 0.338       | 0.345                | 100.0%        |
    | GPT-4.1 (Human)                | 88.2%             | 0.845 / 0.847        | 0.945                 | 0.724 / 0.700        | 0.575                 | 0.559 / 0.534       | 0.508                | 100.0%        |
    | GPT-4.1 Mini (Human)           | 89.7%             | 0.903 / 0.913        | 0.987                 | 0.748 / 0.735        | 0.780                 | 0.554 / 0.510       | 0.763                | 100.0%        |
    | GPT-4.1 Nano (Human)           | 83.5%             | 0.837 / 0.842        | 0.904                 | 0.657 / 0.654        | 0.613                 | 0.380 / 0.306       | 0.229                | 100.0%        |
    | GPT-4o-24-05-13 (Human)        | 89.8%             | 0.864 / 0.868        | 0.967                 | 0.692 / 0.646        | 0.839                 | 0.513 / 0.454       | 0.743                | 100.0%        |
    | GPT-o3-25-04-16 (Human)        | 89.1%             | 0.852 / 0.845        | 0.870                 | 0.785 / 0.767        | 0.796                 | 0.507 / 0.487       | 0.469                | 100.0%        |
    | GPT-o4-mini-25-04-16 (Human)   | 88.0%             | 0.863 / 0.872        | 0.920                 | 0.666 / 0.663        | 0.727                 | 0.439 / 0.423       | 0.555                | 100.0%        |

## 📂 The structure of the projects
- dataset: FRANK and REALSumm (in JSON format) are located in this folder
- finesure: the code to run our FineSurE method to evaluate the summary generated from language models
- reproduce: the code to reproduce the results of FineSurE in Table 1 and Table 2
- result: Results from FineSurE's faithfulness-evaluation and completeness/conciseness-evaluation are saved are located in this folder

## ⚒️ Modification from the [Original FineSurE Repository](https://github.com/DISL-Lab/FineSurE-ACL24)
From the original repository, I have improved upon some points that were missing.
I have modified `reproduce/reproduce-main-results.py` so that it can take multiple parameters for different model settings
I have added codes w.r.t. keyfact extraction with LLM, for this part was omitted from original repo.
I have followed the original FineSurE Paper's settings in doing so.
I have made various improvements on error-checking, reusability, up-to-dateness of api-related codes, etc.

## ⚠️ NOTE on Future-Works
~~From what I've understood, it seems like the LLM-extracted KeyFact list based FineSurE pipeline is somewhat flawed or missing some component as of current implementation.~~
~~(The number of extracted KeyFacts must match that of Human written KeyFacts for the same datum sample within realsumm for the current implementation to correctly calculate correlations between LLM Scores and Human Scores. This is not the case in practice, of course, and I will update the codebase to account for this problem)~~

~~If you do not want to run and skip LLM-extracted KeyFact List based Alignment process, you can simply comment-out the LLM-extracted KeyFact Alignment section within `run_all.sh`~~
~~Below are the results from brute-forcing the codebase to run FineSurE and correlation calculations on LLM-extracted KeyFacts.~~
~~(I've modified rank correlation function so that When KeyFact lengths mismatch leads to empty score lists, it defaults to 0 score)~~
~~As one could point out, the Pearson/Spearman/rank Correlations are skewed or erronous when running correlation calculation against human ground truth labels. (due to the different length of each keyfact lists)~~

`25.06.05` : Modified Completeness / Conciseness Correlation related codes in order to make LLM-Extracted KeyFact List Work, too. (Table beneath represents re-run statistics) (still need to be careful when taking these stats at face value)
| 모델 (KeyFact 출처)                | Faithfulness bAcc | Faithfulness (P / S) | Faithfulness RankCorr | Completeness (P / S) | Completeness RankCorr | Conciseness (P / S) | Conciseness RankCorr | Success Ratio |
| ------------------------------ | ----------------- | -------------------- | --------------------- | -------------------- | --------------------- | ------------------- | -------------------- | ------------- |
| GPT-3.5 Turbo (Machine)        | 67.1%             | 0.668 / 0.720        | 0.898                 | 0.129 / 0.122        | 0.260                 | 0.224 / 0.180       | 0.283                | 100.0%        |
| GPT-4 1106 (Machine)           | 88.4%             | 0.874 / 0.890        | 0.902                 | 0.273 / 0.246        | 0.389                 | 0.274 / 0.257       | 0.618                | 100.0%        |
| GPT-4.1 (Machine)              | 88.2%             | 0.845 / 0.847        | 0.945                 | 0.268 / 0.248        | 0.345                 | 0.177 / 0.176       | 0.290                | 100.0%        |
| GPT-4.1 Mini (Machine)         | 89.7%             | 0.903 / 0.913        | 0.987                 | 0.216 / 0.169        | 0.079                 | 0.144 / 0.148       | -0.031               | 100.0%        |
| GPT-4.1 Nano (Machine)         | 83.5%             | 0.837 / 0.842        | 0.904                 | -0.132 / -0.169      | 0.215                 | 0.076 / 0.062       | 0.149                | 100.0%        |
| GPT-4o-24-05-13 (Machine)      | 89.8%             | 0.864 / 0.868        | 0.967                 | 0.182 / 0.149        | 0.317                 | 0.180 / 0.122       | 0.321                | 100.0%        |
| GPT-o3-25-04-16 (Machine)      | 89.1%             | 0.852 / 0.845        | 0.870                 | 0.043 / -0.015       | -0.101                | 0.119 / 0.051       | 0.228                | 100.0%        |
| GPT-o4-mini-25-04-16 (Machine) | 88.0%             | 0.863 / 0.872        | 0.920                 | 0.273 / 0.236        | 0.433                 | 0.250 / 0.261       | 0.477                | 100.0%        |

## ✅ Highlight

**FineSurE** is a *multi-dimensional*, *fine-grained* automated evaluation framework for text summarization. It covers there distinctive evaluation dimensions, namely faithfulness, completeness, and conciseness. These dimensions are crucial to assess the capability of modern language models in summarization, as they are susceptible to incorrect statement, information omission, and verbosity.

FineSurE framework breaks down a complicate evaluation process into ```two simple human-like evaluation tasks``` using LLMs. 

- **Fact Checking:** This is a task of solving a categorization problem involving nine categories. These include the seven factuality errors, along with an additional category "other error" for error outside the seven errors, and an additional category "no error" for cases whether no error was detected. Given a pair of input text and model summary, the LLM is expected to output the error type classified into one of the nine categories for each sentence along with a concise reason.
  
- **Keyfact Alignment:** This is an alignment task of matching each keyfact into the summary sentences from which the keyfact is inferable. Given a pari of keyfact list and model summary, the output should be the binary label (whether inferable or not) an dth elist of line numbers of all summary sentences matched for each keyfact.

<p align="center">
<img width="755" alt="스크린샷 2024-07-02 오후 4 56 27" src="https://github.com/DISL-Lab/FineSurE-ACL24/assets/10972556/e5c733b7-d863-4e39-92ac-98f63e8bbee5">
</p>

## 🙌🏻 Citation
```BibTeX
@inproceedings{song2024finesure,
  title={FineSurE: Fine-grained Summarization Evaluation using LLMs},
  author={Song, Hwanjun and Su, Hang and Shalyminov, Igor and Cai, Jason and Mansour, Saab},
  booktitle={ACL},
  year={2024}
}
```

## ✉️ Contact
Feel free to post issues on github, or contact my e-mail (wjkim9653@gmail.com)