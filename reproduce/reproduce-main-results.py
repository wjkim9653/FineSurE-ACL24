import os
import sys
import json
import numpy as np
from utils import compute_faithfulness_percentage_score, compute_completeness_percentage_score
from scipy.stats import pearsonr, spearmanr
import scipy.stats as ss
import logging
logging.basicConfig(level=logging.WARNING)

def main(frank_result_path, realsumm_result_path, model, keyfact):
    with open(os.path.join('reproduce/logs/',f'{model}-{keyfact}.txt'), 'w') as logfile:
        # 1. load frank data
        frank_results = []
        for line in open(frank_result_path, 'r'):
            line = json.loads(line)
            frank_results.append(line)

        faithfulness_eval(frank_results, logfile)

        # load realsumm data
        realsumm_results = []
        for line in open(realsumm_result_path, 'r'):
            line = json.loads(line)
            realsumm_results.append(line)
            
        completeness_and_conciseness_eval(realsumm_results, logfile)


def faithfulness_eval(results: list, logfile):
    '''
    A function to evaluate the results from FineSurE on faithfulness at the three different levels
    '''

    # faithfulness eval
    model_wise_results = {}
    conv_ids = []
    task_keys = []
    summary_keys = []
    full_results = {
        # general factaulity (sentence-level)¸¸
        'gt_faithfulness_binary_labels': [],
        'pred_faithfulness_binary_labels': [],
        # general factaulity (summary-level)
        'gt_faithfulness_scores': [],
        'pred_faithfulness_scores': [],
    }

    cnt_success_inference = 0

    for result in results:
        conv_id = result['doc_id']
        dataset_name = result['source']
        model = result['model']

        # dict for computing the overall scores, and system-wise ranking
        if model not in model_wise_results:
            model_wise_results[model] = {
                'gt_faithfulness_binary_labels': [],
                'pred_faithfulness_binary_labels': [],
                'gt_faithfulness_scores': [],
                'pred_faithfulness_scores': [],
            }
    
        conv_ids.append(conv_id + model)

        # get gt labels and pred labels
        gt_faithfulness_binary_labels = get_aggregate_gt_labels(result['raw_annotations'], key="factuality_labels")  # 문장수(라벨수)만큼 담긴 리스트로 반환됨
        pred_faithfulness_binary_labels = result['pred_faithfulness_labels']  # 문장수(라벨수)만큼 담긴 리스트로 반환됨

        if len(gt_faithfulness_binary_labels) != len(pred_faithfulness_binary_labels):
            # failure cases → 문장수와 상응 라벨수가 동일하지 않는 경우에 해당
            continue

        _gt_faithfulness_binary_labels, _pred_faithfulness_binary_labels = [], []
        for idx, item in enumerate(gt_faithfulness_binary_labels):
            # exception handler
            if item != 'None':  # 앞서 gt_faithfulness_binary_labels 구하기 위해 get_aggregate_gt_labels() 함수 실행할 때, 특정 요약문장에 대한 annotator label이 누락된 엣지케이스 대응해야 하므로 조건문
                _gt_faithfulness_binary_labels.append(gt_faithfulness_binary_labels[idx])
                _pred_faithfulness_binary_labels.append(pred_faithfulness_binary_labels[idx])
        gt_faithfulness_binary_labels, pred_faithfulness_binary_labels = np.array(_gt_faithfulness_binary_labels), np.array(_pred_faithfulness_binary_labels)
        
        if len(gt_faithfulness_binary_labels) == 0:
            # failure cases
            continue

        key = dataset_name + '-' + conv_id + '-' + model
        for sentence_id in range(len(gt_faithfulness_binary_labels)):
            task_keys.append(key + '-' + str(sentence_id + 1))
        summary_keys.append(key)

        cnt_success_inference += 1
  
        #### compute summary-level
        gt_faithfulness_score = compute_faithfulness_percentage_score(gt_faithfulness_binary_labels)
        pred_faithfulness_score = compute_faithfulness_percentage_score(pred_faithfulness_binary_labels)

        full_results['gt_faithfulness_binary_labels'].extend(gt_faithfulness_binary_labels)
        full_results['pred_faithfulness_binary_labels'].extend(pred_faithfulness_binary_labels)
        full_results['gt_faithfulness_scores'].append(gt_faithfulness_score)
        full_results['pred_faithfulness_scores'].append(pred_faithfulness_score)

        model_wise_results[model]['gt_faithfulness_binary_labels'].extend(gt_faithfulness_binary_labels)
        model_wise_results[model]['pred_faithfulness_binary_labels'].extend(pred_faithfulness_binary_labels)
        model_wise_results[model]['gt_faithfulness_scores'].append(gt_faithfulness_score)
        model_wise_results[model]['pred_faithfulness_scores'].append(pred_faithfulness_score)

    logging.info('[Faithfulness Evaluation]')
    logfile.write('[Faithfulness Evaluation]\n')

    logging.info('* Sentence-level')
    logfile.write('* Sentence-level\n')
    bAcc = balancedAcc(full_results['gt_faithfulness_binary_labels'], full_results['pred_faithfulness_binary_labels'])
    logging.info(f'\t-Balanced Accuracy:{bAcc:.1%}')
    logfile.write(f'\t-Balanced Accuracy:{bAcc:.1%}\n')

    pearson_corr = pearsonr(full_results['gt_faithfulness_scores'], full_results['pred_faithfulness_scores'])
    spearman_corr = spearmanr(full_results['gt_faithfulness_scores'], full_results['pred_faithfulness_scores'])

    logging.info('* Summary-level:')
    logging.info(f"\t-Pearson: {pearson_corr}")
    logging.info(f"\t-Spearman: {spearman_corr}")
    
    logfile.write('* Summary-level:\n')
    logfile.write(f"\t-Pearson: {pearson_corr}\n")
    logfile.write(f"\t-Spearman: {spearman_corr}\n")

    logging.info('* System-level:')
    logfile.write('* System-level:\n')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="faithfulness_scores")
    logging.info(f"\t-Rank Correlation: {_rank_correlation}")
    logfile.write(f"\t-Rank Correlation: {_rank_correlation}\n")

    success_rate = cnt_success_inference / len(conv_ids)
    logging.info(f'* Success ratio {success_rate:.1%}')
    logfile.write(f'* Success ratio {success_rate:.1%}\n')


def completeness_and_conciseness_eval(results, logfile):
    '''
    A function to evaluate the results from FineSurE on faithfulness at the three different levels
    '''

    # faithfulness eval
    model_wise_results = {}
    conv_ids = []
    task_keys = []
    summary_keys = []
    full_results = {
        'gt_completeness_scores': [],
        'pred_completeness_scores': [],
        'gt_conciseness_scores': [],
        'pred_conciseness_scores': [],
    }

    cnt_success_inference = 0

    for result in results:
        conv_id = result['doc_id']
        dataset_name = result['source']
        model = result['model']

        # dict for computing the overall scores, and system-wise ranking
        if model not in model_wise_results:
            model_wise_results[model] = {
                'gt_completeness_scores': [],
                'pred_completeness_scores': [],
                'gt_conciseness_scores': [],
                'pred_conciseness_scores': [],
            }
    
        conv_ids.append(conv_id + model)

        # get gt labels and pred labels
        gt_alignment_labels = get_aggregate_gt_labels(result['raw_annotations'], key="key_fact_labels")
        gt_sentence_line_numbers = get_aggregate_gt_labels(result['raw_annotations'], key="sentence_labels")
        pred_alignment_labels = result['pred_alignment_labels']
        pred_sentence_line_numbers = result['pred_sentence_line_numbers']

        # failure cases
        if len(gt_alignment_labels) != len(pred_alignment_labels):
            continue
        _gt_alignment_labels, _pred_alignment_labels = [], []
        for idx, item in enumerate(gt_alignment_labels):
            if item != 'None':
                _gt_alignment_labels.append(gt_alignment_labels[idx])
                _pred_alignment_labels.append(pred_alignment_labels[idx])
        gt_alignment_labels, pred_alignment_labels = np.array(_gt_alignment_labels), np.array(_pred_alignment_labels)

        key = dataset_name + '-' + conv_id + '-' + model
        for key_fact_id in range(len(gt_alignment_labels)):
            task_keys.append(key + '-' + str(key_fact_id + 1))
        summary_keys.append(key)

        cnt_success_inference += 1

        # compute completeness percentage
        gt_completeness_score = compute_completeness_percentage_score(gt_alignment_labels)
        pred_completeness_score = compute_completeness_percentage_score(pred_alignment_labels)

        # compute conciseness percentage
        _pred_sentence_line_numbers = []
        for idx in range(len(gt_sentence_line_numbers)):
            if (idx +1) in pred_sentence_line_numbers:
                _pred_sentence_line_numbers.append(1.0)
            else:
                _pred_sentence_line_numbers.append(0.0)
        pred_sentence_line_numbers = _pred_sentence_line_numbers

      
        assert len(gt_sentence_line_numbers) == len(pred_sentence_line_numbers)
        gt_conciseness_score = sum(gt_sentence_line_numbers) / len(gt_sentence_line_numbers)
        pred_conciseness_score = sum(pred_sentence_line_numbers) / len(pred_sentence_line_numbers)

        full_results['gt_completeness_scores'].append(gt_completeness_score)
        full_results['pred_completeness_scores'].append(pred_completeness_score)
        full_results['gt_conciseness_scores'].append(gt_conciseness_score)
        full_results['pred_conciseness_scores'].append(pred_conciseness_score)

        model_wise_results[model]['gt_completeness_scores'].append(gt_completeness_score)
        model_wise_results[model]['pred_completeness_scores'].append(pred_completeness_score)
        model_wise_results[model]['gt_conciseness_scores'].append(gt_conciseness_score)
        model_wise_results[model]['pred_conciseness_scores'].append(pred_conciseness_score)

    logging.info('\n[Completeness Evaluation]')
    logfile.write('\n[Completeness Evaluation]\n')

    logging.info('* Summary-level:')
    logfile.write('* Summary-level:\n')
    try:
        pearson_corr = pearsonr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])
        logging.info(f"\t-Pearson: {pearson_corr}")
        logfile.write(f"\t-Pearson: {pearson_corr}\n")
    except Exception as e:
            logging.info(f'Failed to Calculate Pearson Corr. : {e}')
            logfile.write(f'Failed to Calculate Pearson Corr. : {e}')
    try:
        spearman_corr = spearmanr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])
        logging.info(f"\t-Spearman: {spearman_corr}")
        logfile.write(f"\t-Spearman: {spearman_corr}\n")
    except Exception as e:
        logging.info(f'Failed to Calculate Spearman Corr. : {e}')
        logfile.write(f'Failed to Calculate Spearman Corr. : {e}')


    logging.info('* System-level:')
    logfile.write('* System-level:\n')
    # model-wise ranking 
    try:
        _rank_correlation = rank_correlation(model_wise_results, key="completeness_scores")
        logging.info(f"\t-Rank Correlation: {_rank_correlation}")
        logfile.write(f"\t-Rank Correlation: {_rank_correlation}\n")
    except Exception as e:
        logging.info(f'Failed to Calculate Rank Corr. : {e}')
        logfile.write(f'Failed to Calculate Rank Corr. : {e}')



    logging.info('\n[Conciseness Evaluation]')
    logfile.write('\n[Conciseness Evaluation]\n')
    logging.info('* Summary-level:')
    logfile.write('* Summary-level:\n')
    try:
        pearson_corr = pearsonr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])
        logging.info(f"\t-Pearson: {pearson_corr}")
        logfile.write(f"\t-Pearson: {pearson_corr}\n")
    except Exception as e:
        logging.info(f'Failed to Calculate Pearson Corr. : {e}')
        logfile.write(f'Failed to Calculate Pearson Corr. : {e}')

    try:
        spearman_corr = spearmanr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])
        logging.info(f"\t-Spearman: {spearman_corr}")
        logfile.write(f"\t-Spearman: {spearman_corr}\n")
    except Exception as e:
        logging.info(f'Failed to Calculate Spearman Corr. : {e}')
        logfile.write(f'Failed to Calculate Spearman Corr. : {e}')

    logging.info('* System-level:')
    logfile.write('* System-level:\n')
    # model-wise ranking 
    try:
        _rank_correlation = rank_correlation(model_wise_results, key="conciseness_scores")
        logging.info(f"\t-Rank Correlation: {_rank_correlation}")
        logfile.write(f"\t-Rank Correlation: {_rank_correlation}\n")
    except Exception as e:
        logging.info(f'Failed to Calculate Rank Corr. : {e}')
        logfile.write(f'Failed to Calculate Rank Corr. : {e}')


    success_rate = cnt_success_inference / len(conv_ids)
    logging.info(f'\n* Success ratio {success_rate:.1%}')
    logfile.write(f'\n* Success ratio {success_rate:.1%}\n')
  

def get_aggregate_gt_labels(raw_annotations:dict, key:str):
    '''
    A function to generate the aggregated human labels from three annotators
    Args:
        - raw_annotations: the raw annotations from three annotators
        - key: the annotation type ('xxx:' )
    Returns:
        - final_labels: the aggregated labels by majority voting
    '''

    # if there are four annotators, we remove the last
    if key == "sentence_labels" and "3" in raw_annotations:
        del raw_annotations["3"]

    merged_gt_labels = []
    for worker_id, annotation in raw_annotations.items():
        gt_labels = annotation[key]  # type(gt_labels): list (e.g. [1])
        merged_gt_labels.append(gt_labels)  # type(merged_gt_labels): list of lists (e.g. [[1], [1], [0]])

    final_labels = []
    merged_gt_labels = np.array(merged_gt_labels)  # 2차원 numpy array
    num_labels = len(merged_gt_labels[-1])  # 문장수(상응라벨수)가 1개 넘는 경우도 있으므로 (주석 예시는 문장수 1개를 상정함)

    for sent_idx in range(num_labels):
        _column = merged_gt_labels[:, sent_idx]  # e.g. → array([1, 1, 0])  ← 각 annotator가 그 문장에 준 라벨
        _column = [float(item) for item in _column if item != 'None']  # e.g. → [1.0, 1.0, 0.0]

        if len(_column) <= 1:
            final_labels.append('None')
        else:
            final_label = max(set(_column), key = _column.count)  # 가장 많이 등장한 값을 final_label로 (e.g. 위 예시대로면 1.0이 final label)
            final_labels.append(float(final_label))

    assert len(final_labels) == num_labels

    return final_labels  # gt annotation에서 가장 많이 등장한 값을 담은 리스트 (e.g. [1.0]) 반환


def balancedAcc(gt, pred):
    '''
    A function to compute the balanced accuracy
    Args:
        - gt: ground truth labels
        - pred: predicted labels
    Return:
        - balanced accuracy
    '''
    ones, zeros = [], []
    for idx in range(len(gt)):
        if gt[idx] == 1.0:
            ones.append(pred[idx])
        elif gt[idx] == 0.0:
            zeros.append(pred[idx])

    error_acc = sum(ones) / len(ones)
    non_error_acc =  1.0 - sum(zeros) / len(zeros)

    return (error_acc + non_error_acc) / 2.0


def rank_correlation(model_wise_results, key, min_number=5):
    '''
    A function to compute the balanced accuracy
    Args:
        - model_wise_results: evaluation results per model in dict
        - key: evaluation dimension
        - min_number: the minimum number of examples to be included in the evaluation
    Return:
        - rank correlation with p value
    '''

    model_list =  model_wise_results.keys()

    models = []
    gt_errors = []
    pred_errors = []
    for model_name in model_list:
        models.append(model_name)
        gt_error, pred_error = np.mean(model_wise_results[model_name]['gt_' + key]), np.mean(model_wise_results[model_name]['pred_' + key])

        if len(model_wise_results[model_name]['gt_' + key]) >= min_number:
            gt_errors.append(gt_error)
            pred_errors.append(pred_error)

    pred_errors = np.array(pred_errors) 
    gt_errors = np.array(gt_errors) 

    estimated_rank = ss.rankdata(pred_errors)
    human_rank = ss.rankdata(gt_errors)
    #logging.info("models:", models)
    #logging.info('gt ' + key + ':', gt_errors)
    #logging.info('pred ' + key + ':', pred_errors )
    #logging.info('gt rank ' + key + ':', human_rank)
    #logging.info('pred rank ' + key + ':', estimated_rank)
    spearman_corr = spearmanr(estimated_rank, human_rank)

    return spearman_corr



if __name__ == "__main__":

    '''
    Runnining Command:
        (1)
        cd FineSurE
        
        (2) 
        python reproduce/reproduce-main-results.py \
            [keyfact json file directory] \
            [results path]
        e.g.
        python reproduce/reproduce-main-results.py \
            dataset/realsumm/keyfact/ \
            result/
    '''

    keyfact_path = sys.argv[1]
    if not os.path.isdir(keyfact_path):
        raise ValueError("KeyFact Path Not Found")
    
    result_path = sys.argv[2]
    if not os.path.isdir(result_path):
        raise ValueError("Result Path Not Found")
    
    models = [
        'gpt-3.5-turbo',
        'gpt-4-1106-preview',
        'gpt-4o-2024-05-13',
        'gpt-4.1-2025-04-14',
        'gpt-4.1-mini-2025-04-14',
        'gpt-4.1-nano-2025-04-14',
        'o3-2025-04-16',
        'o4-mini-2025-04-16',
    ]

    for model in models:
        if (
            os.path.isfile(os.path.join(keyfact_path, 'human-keyfact-list.json'))
            and
            os.path.isfile(os.path.join(result_path, 'keyfact-alignment', f'realsumm-raw-data-by-{model}-keyfact-from-human-keyfact-list.json'))
        ):  # 만약 인간이 제공한 KeyFact 기반으로 수행한 KeyFact-Alignment 결과 파일이 있을 경우
            main(
                frank_result_path=os.path.join(result_path, 'fact-checking', f'frank-raw-data-by-{model}.json'), 
                realsumm_result_path=os.path.join(result_path, 'keyfact-alignment', f'realsumm-raw-data-by-{model}-keyfact-from-human-keyfact-list.json'),
                model=model,
                keyfact='human-keyfact-list'
            )
        if (
            os.path.isfile(os.path.join(keyfact_path, f'machine-keyfact-list-from-{model}.json'))
            and
            os.path.isfile(os.path.join(result_path, 'keyfact-alignment', f'realsumm-raw-data-by-{model}-keyfact-from-machine-keyfact-list-from-{model}.json'))
        ):  # 만약 해당 모델로 추출한 KeyFact 기반으로 수행한 KeyFact-Alignment 결과 파일이 있을 경우
            main(
                frank_result_path=os.path.join(result_path, 'fact-checking', f'frank-raw-data-by-{model}.json'), 
                realsumm_result_path=os.path.join(result_path, 'keyfact-alignment', f'realsumm-raw-data-by-{model}-keyfact-from-machine-keyfact-list-from-{model}.json'),
                model=model,
                keyfact=f'machine-keyfact-list-from-{model}'
            )