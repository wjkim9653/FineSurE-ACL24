import openai
import json
import sys
import os
from utils import get_response, parsing_llm_fact_checking_output, compute_faithfulness_percentage_score
from utils import get_fact_checking_prompt
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
import random
import logging

load_dotenv()
random.seed(42)
logging.basicConfig(level=logging.WARNING)

# api key
_api_key = os.getenv("OPENAI_API_KEY")
# logging.info(_api_key)
_client = openai.OpenAI(api_key=_api_key)
#_model = "gpt-3.5-turbo"
#_model = "gpt-4-1106-preview"
_model = "gpt-4o-2024-05-13"  # fallback (in case no `models` list is defined when calling main())

def main(input_path, output_path, log_interval=2, model=_model, sample_cnt=100, position=0):
    '''
    Argument:
        input_path: path for input data
        output_path: path for output data (saving the logs and the eval results)
        log_interval: log the percentage scores every 'log_interval'
        model: OpenAI API Compatible Models ID to Test
        sample_cnt: # of Samples from realsumm dataset, must be integer, defaults to 100
    '''
    logging.info(f"✅ Running Fact-Checking with 🤖: {model}")

    # loads data for faithfulness evaluation using FineSurE
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    sampled_inputs = random.sample(inputs, sample_cnt)  # 100개 랜덤샘플링한 데이터셋으로 진행

    # variables for evaluation 
    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    
    # writer to store the output from LLM evaluation
    raw_data_writer = open(os.path.join(output_path, f'frank-raw-data-by-{model}.json'), 'w')
    result_writer = open(os.path.join(output_path, f'frank-result-by-{model}.json'), 'w')

    # processes each data instance using for loop
    for input_id, input_json in enumerate(tqdm(sampled_inputs, desc=f'[#{position}] {model}', position=position, leave=False)):

        # input json parsing
        doc_id = input_json['doc_id']
        model_name = input_json['model']
        src = input_json['transcript']
        sentences = input_json['sentences']

        # prompt generation
        prompt = get_fact_checking_prompt(input=src, sentences=sentences)

        # get response from LLMs
        output = get_response(client=_client, prompt=prompt, model=model)
 
        input_json['llm_output'] = output
        input_json['pred_faithfulness_labels'], input_json['pred_faithfulness_error_type'] = parsing_llm_fact_checking_output(output)

        # check if the parsing is success
        success_flag = True
        if len(input_json['pred_faithfulness_labels']) == 0:
            success_flag = False

        logging.info("\nInput ID:", doc_id, "Model Name:", model_name)
        logging.info("Success:", success_flag)
        logging.info('\t[Error Label]:', input_json['pred_faithfulness_labels'])
        logging.info('\t[Error Type]:', input_json['pred_faithfulness_error_type'])

        # count the success cases
        cnt_total_inference += 1
        if success_flag:
            cnt_success_inference += 1
        else:
            # fail to evalaute -> skip
            continue

        # compute the percentage score for faithfulness
        faithfulness_score = compute_faithfulness_percentage_score(input_json['pred_faithfulness_labels'])
        logging.info('\t[Faithfulness Score]:', '{:.1%}'.format(faithfulness_score))

        # put the score into the aggregation dictionary
        if model_name not in model_labels:
            model_labels[model_name] = {'faithfulness_scores': [], 'binary_labels': []}
        model_labels[model_name]['faithfulness_scores'].append(faithfulness_score)
        model_labels[model_name]['binary_labels'].extend(input_json['pred_faithfulness_labels'])

        def log_results_faithfulness(model_labels):
            sentence_level_errors = {}
            summary_level_scores = {}
            for model_name, error_labels in model_labels.items():
                sentence_level_errors[model_name] = sum(error_labels['binary_labels']) / len(error_labels['binary_labels'])
                summary_level_scores[model_name] = sum(error_labels['faithfulness_scores']) / len(error_labels['faithfulness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '* sentence-level factuality error ratio per model (lower is better)\n'
            for model_name, error_rate in sentence_level_errors.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(error_rate)) + '\n'

            text_output += '\n* summary-level faithfulness score per model (higher is better)\n'
            for model_name, score in summary_level_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* system-level model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            success_ratio = '{:.1%}'.format(cnt_success_inference/float(cnt_total_inference))
            text_output += '\n* success rate: ' + str(success_ratio) + '\n\n\n'

            logging.info(text_output)
            return text_output

        # log percentage score
        if cnt_total_inference % log_interval == 0:
            log_results_faithfulness(model_labels=model_labels)
           
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
    raw_data_writer.close()

    # log final results
    text_output = log_results_faithfulness(model_labels=model_labels)
    result_writer.write(text_output)


def run_per_model(args):
    input_path, output_path, log_interval, model, sample_count = args
    position = MODEL_POSITION_MAP.get(model, 0)
    main(input_path, output_path, log_interval, model, sample_count, position)
    return model  # main() 성공적으로 실행 끝난 모델명 반환

if __name__ == "__main__":
    
    '''
    Runnining Command:
        1) cd FineSurE-ACL24
        2) python finesure/fact-checking.py [input-path] [output-folder] [# of samples to run]
        e.g., 
        python finesure/fact-checking.py \
            dataset/frank/frank-data.json \
            result/fact-checking \
            100
    '''

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    sample_cnt = int(sys.argv[3])
    log_interval = 10  # log logs every 10 inferences

    models = [
        'gpt-3.5-turbo',
        'gpt-4-1106-preview',
        'gpt-4o-2024-05-13',
        'gpt-4.1-2025-04-14',
        'gpt-4.1-mini-2025-04-14',
        'gpt-4.1-nano-2025-04-14',
        'o3-2025-04-16',
        'o4-mini-2025-04-16'
    ]

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # 각 모델별로 병렬 실행하기 위한 run_per_model() arguments
    MODEL_POSITION_MAP = {model: i+1 for i, model in enumerate(models)}
    args_list = [(input_path, output_path, log_interval, model, sample_cnt) for model in models]
    max_workers = 8

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(run_per_model, args_list), total=len(models), desc=f"✅ Fact-Checking for {len(models)} Models Running in Parallel w/ {max_workers} workers"))