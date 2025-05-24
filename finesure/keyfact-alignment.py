import openai
import json
import sys
import os
from utils import get_response
from utils import get_keyfact_alighment_prompt, parsing_llm_keyfact_alighment_output
from utils import compute_completeness_percentage_score, compute_conciseness_percentage_score
from utils import get_keyfact_extraction_prompt, keyfact_extraction
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)
import random

# api key
_api_key = os.getenv("OPENAI_API_KEY")
_client = openai.OpenAI(api_key=_api_key)
#_model = "gpt-3.5-turbo"
#_model = "gpt-4-1106-preview"
_model = "gpt-4o-2024-05-13"  # fallback (in case no `models` list is defined when calling main())

def main(input_path, keyfact_path, output_path, print_interval=2, model=_model, sample_cnt=100):
    '''
    Argument:
        input_path: path for input data
        keyfact_path: path for human or machine keyfacts
        output_path: path for output data (saving the logs and the eval results)
        print_interval: print the percentage scores every 'print_interval' 
        model: OpenAI API Compatible Models ID to Test
        sample_cnt: # of Samples from realsumm dataset, must be integer, defaults to 100
    '''
    logging.info(f"🤖 Running KeyFact-Alignment with Model: {model}")

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
            json.dump({}, f)
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
                new_entry['keyfacts'] = keyfacts
                with open(keyfact_path, 'r') as f:
                    f.write(json.dumps(new_entry, ensure_ascii=False)+'\n')
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
    raw_data_writer = open(os.path.join(output_path, f'realsumm-raw-data-by-{model}.json'), 'w')
    result_writer = open(os.path.join(output_path, f'realsumm-result-by-{model}.json'), 'w')

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

        print("\nInput ID:", doc_id, "Model Name:", model_name)
        print("Success:", success_flag)
        print('\t[Alignment Label]:', input_json['pred_alignment_labels'])
        print('\t[Matched Sentence Line Numbers]:', input_json['pred_sentence_line_numbers'])

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

        print('\t[Completeness Score]:', '{:.1%}'.format(completeness_score))
        print('\t[Conciseness Score]:', '{:.1%}'.format(conciseness_score))

        def print_results_faithfulness(model_labels):
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

            print(text_output)
            return text_output

        # print percentage score
        if cnt_total_inference % print_interval == 0:
            print_results_faithfulness(model_labels=model_labels)
           
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
    raw_data_writer.close()

    # print final results
    text_output = print_results_faithfulness(model_labels=model_labels)
    result_writer.write(text_output)
           

if __name__ == "__main__":
    
    '''
    Runnining Command:
        1) cd CodeRelease
        
        2) python finesure/keyfact-alignment.py [input-path] [keyfact-path] [output-folder]
        e.g., python finesure/keyfact-alignment.py dataset/realsumm/realsumm-data.json dataset/realsumm/human-keyfact-list.json result/keyfact-alignment
    '''

    input_path = sys.argv[1]
    keyfact_path = sys.argv[2]
    output_folder = sys.argv[3]
    sample_cnt = sys.argv[4]

    # print logs every 10 inferences
    print_interval = 10

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

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for model in models:
        main(input_path, keyfact_path, output_folder, print_interval, model, sample_cnt)
