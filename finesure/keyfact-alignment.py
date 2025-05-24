import openai
import json
import sys
import os
import random
from utils import get_response
from utils import get_keyfact_alighment_prompt, parsing_llm_keyfact_alighment_output
from utils import compute_completeness_percentage_score, compute_conciseness_percentage_score
from utils import get_keyfact_extraction_prompt, keyfact_extraction
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)

# api key
_api_key = os.getenv("OPENAI_API_KEY")
_client = openai.OpenAI(api_key=_api_key)

def main(input_path, keyfact_path, output_path, log_interval=2, model='gpt-4o-2024-05-13', sample_cnt=100):
    '''
    Argument:
        input_path: path for input data
        keyfact_path: path for human or machine keyfacts
        output_path: path for output data (saving the logs and the eval results)
        log_interval: log the percentage scores every 'log_interval' 
        model: OpenAI API Compatible Models ID to Test
        sample_cnt: # of Samples from realsumm dataset, must be integer, defaults to 100
    '''
    logging.info(f"✅ Running KeyFact-Alignment with 🤖: {model} | KeyFact File from 📄: {keyfact_path}")

    # loads data for completeness and conciseness evaluation using FineSurE
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)
    
    sampled_inputs = random.sample(inputs, sample_cnt)  # 100개 랜덤샘플링한 데이터셋으로 진행
    
    # loads keyfacts
    if os.path.isfile(keyfact_path):  # 이미 추출된 KeyFact가 전달받은 json파일 경로로 존재하는 경우
        logging.info(f"Found existing Machine-KeyFact-Extraction-Json File: {keyfact_path}")
    else:  # 아직 KeyFact 추출 json이 존재하지 않는 경우
        with open(keyfact_path, 'w') as f:
            pass
        logging.info(f"Created a New Machine-KeyFact-Extraction-Json File: {keyfact_path}")

    # json에 저장되어 있는 KeyFact 추출 결과가 완전하지 않은 경우 대응 (sampled_inputs의 docid 중 keyfact_path 파일에 누락된 경우 캐치해 api call & write)
    with open(keyfact_path, 'r') as f:
        existing_keyfact = [json.loads(line) for line in f]
    for sample_input in sampled_inputs:
        found = any(entry.get('doc_id')==sample_input['doc_id'] for entry in existing_keyfact)
        if not found:  # Machine Key Fact 추출이 누락된 경우
            logging.info(f"Found missing Machine-KeyFact for doc_id: {sample_input['doc_id']} | Calling OpenAI API")
            keyfact_extraction_prompt = get_keyfact_extraction_prompt(sample_input['sentences'])
            try:
                llm_output, keyfacts = keyfact_extraction(
                    client=_client,
                    prompt=keyfact_extraction_prompt,
                    model=model,
                )  # OpenAI API Call 통해 Machine-Extracted-KeyFacts 확보
                
                # `keyfact_path` file에 추가
                new_entry = sample_input.copy()
                new_entry['llm_output'] = llm_output
                new_entry['key_facts'] = keyfacts
                with open(keyfact_path, 'a') as f:
                    f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                logging.info(f"Successfully Saved Newly Extracted Machine-KeyFact for doc_id: {sample_input['doc_id']}")
            except Exception as e:
                logging.error(f"KeyFact Extraction Failed :\n{e}")

    # KeyFact 읽어오기
    keyfacts = {}
    for line in open(keyfact_path, 'r'):
        line = json.loads(line)
        keyfacts[line['doc_id']] = line['key_facts']

    # variables for evaluation
    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    
    # writer to store the output from LLM evaluation
    raw_data_writer = open(os.path.join(output_path, f'realsumm-raw-data-by-{model}-keyfact-from-{keyfact_path}.json'), 'w')
    result_writer = open(os.path.join(output_path, f'realsumm-result-by-{model}-keyfact-from-{keyfact_path}.json'), 'w')

    # processes each data instance using for loop
    for input_id, input_json in enumerate(sampled_inputs):

        # input json parsing
        doc_id = input_json['doc_id']
        model_name = input_json['model']
        src = input_json['transcript']
        sentences = input_json['sentences']
        list_keyfacts = keyfacts[doc_id]

        # prompt generation
        prompt = get_keyfact_alighment_prompt(keyfacts=list_keyfacts, sentences=sentences)

        # get response from LLMs
        output = get_response(client=_client, prompt=prompt, model=model)
 
        input_json['llm_output'] = output
        input_json['pred_alignment_labels'], input_json['pred_sentence_line_numbers'] = parsing_llm_keyfact_alighment_output(output)

        # check if the parsing is success
        success_flag = True
        if len(input_json['pred_alignment_labels']) == 0:
            success_flag = False

        logging.info(f"\nInput ID: {doc_id}, Model Name: {model_name}")
        logging.info(f"Success: {success_flag}")
        logging.info(f"\t[Alignment Label]: {input_json['pred_alignment_labels']}")
        logging.info(f"\t[Matched Sentence Line Numbers]: {input_json['pred_sentence_line_numbers']}")

        # count the success cases
        cnt_total_inference += 1
        if success_flag:
            cnt_success_inference += 1
        else:
            # fail to evalaute -> skip
            continue      

        # compute the percentage score for faithfulness
        completeness_score = compute_completeness_percentage_score(input_json['pred_alignment_labels'])
        conciseness_score = compute_conciseness_percentage_score(input_json['pred_sentence_line_numbers'], len(sentences))

        # put the score into the aggregation dictionary
        if model_name not in model_labels:
            model_labels[model_name] = {'completeness_scores': [], 'conciseness_scores': []}
        model_labels[model_name]['completeness_scores'].append(completeness_score)
        model_labels[model_name]['conciseness_scores'].append(conciseness_score)

        logging.info(f'\t[Completeness Score]: {completeness_score:.1%}')
        logging.info(f'\t[Conciseness Score]: {conciseness_score:.1%}')

        def log_results_faithfulness(model_labels):
            summary_level_completeness_scores = {}
            summary_level_conciseness_scores = {}

            for model_name, error_labels in model_labels.items():
                summary_level_completeness_scores[model_name] = sum(error_labels['completeness_scores']) / len(error_labels['completeness_scores'])
                summary_level_conciseness_scores[model_name] = sum(error_labels['conciseness_scores']) / len(error_labels['conciseness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '\n* completeness score per model (higher is better)\n'
            for model_name, score in summary_level_completeness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* completeness model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_completeness_scores.items(), key=lambda item: item[1], reverse=True))
            model_ranking = list(sorted_dict.keys())
            text_output += str(model_ranking) + '\n'

            text_output += '\n* conciseness score per model (higher is better)\n'
            for model_name, score in summary_level_conciseness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* conciseness model ranking (left is better)\n'
            sorted_dict = dict(sorted(summary_level_conciseness_scores.items(), key=lambda item: item[1], reverse=True))
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
           

if __name__ == "__main__":
    
    '''
    Runnining Command:
        1) cd FineSurE-ACL24
        2) python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder] [# of samples to run]
        e.g. (Human Written KeyFact 기반 평가 진행하고 싶은 경우 -> [keyfact-path] 인자로 구체적인 .json 파일을 지정하면 됨)
        python finesure/keyfact-alignment.py \
            dataset/realsumm/realsumm-data.json \
            dataset/realsumm/keyfact/human-keyfact-list.json \
            result/keyfact-alignment \
            10
        
        e.g. (Model-Extracted KeyFact 기반 평가 진행하고 싶은 경우 -> [keyfact-path] 인자로 디렉토리 경로만 지정하면 됨])
        python finesure/keyfact-alignment.py \
            dataset/realsumm/realsumm-data.json \
            dataset/realsumm/keyfact/ \
            result/keyfact-alignment \
            10
    '''

    input_path = sys.argv[1]
    keyfact_path = sys.argv[2]
    output_folder = sys.argv[3]
    sample_cnt = int(sys.argv[4])
    
    # log logs every 10 inferences
    log_interval = 10
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

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

    models_keyfacts_config = {
        model: os.path.join(keyfact_path, f'machine-keyfact-list-from-{model}.json') if os.path.isdir(keyfact_path) else keyfact_path
        for model in models
    }
    
    for model, keyfact_path in models_keyfacts_config.items():
        main(input_path, keyfact_path, output_folder, log_interval, model, sample_cnt)
