'''
This is for providing fundamental functions for FineSurE.
'''
import ast

ERROR_TYPES = [
    'out-of-context error', 
    'entity error', 
    'predicate error', 
    'circumstantial error', 
    'grammatical error', 
    'coreference error', 
    'linking error', 
    'other error'
]

def get_response(client, prompt, model, temperature=0.0):

    ''' A function to get the response from GPT-series
    Args:
        client: openai client
        prompt: input prompt
        model: openai model name
    Return: 
        text_response: the output from LLMs
    '''

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text_response = response.choices[0].message.content

    return text_response



'''
Two functions for fact checking
'''
def get_fact_checking_prompt(input, sentences):

    ''' A function to define the input prompt
    Args:
        input: input document
        sentences: list of summary sentences
    Return: 
        prompt: the final input prompt
    '''

    num_sentences = str(len(sentences))
    sentences = '\n'.join(sentences)

    prompt = \
"""
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":
[{"sentence": "first sentence", "reason": "your reason", "category": "no error"}, {"sentence": "second sentence", "reason": "your reason", "category": "out-of-context error"}, {"sentence": "third sentence", "reason": "your reason", "category": "entity error"},]

Transcript:
%s

Summary with %s sentences:
%s
""" % (input, num_sentences, sentences)

    return prompt


def parsing_llm_fact_checking_output(output):

    ''' A function to parse the output from LLMs based on heuristic rules
    Args:
        output: the output from LLMs
    Return: 
        pred_labels: the binary label for each sentence (0: no factuality error, 1: factuality error)
        pred_types: the error type of each sentence 
    '''

    try:
        start_idx = output.find('[')

        if start_idx != -1:  # 1개 이상 json 담긴 list로 출력된 경우
            end_idx = output.find(']')
            output = output[start_idx:end_idx+1]  # Type은 그대로 str, 단 list of dict 형태 제외한 여타 출력부 제거됨
            output = output.replace('\n','')  # LLM 출력에 포함되어있을 수 있는 개행문자 삭제
            output = ast.literal_eval(output)  # list of dict 형태의 str 타입인 output을 실제 파이썬 list 객체로 변환

            pred_labels, pred_types = [], []
            for out in output:
                category = out["category"]
                category = category.replace('\n', '').replace('[', '').replace(']', '')  # 모델 출력에서 불필요한 개행문자나 리스트 브래킷 포함되는 경우가 있었나봄
                if category.lower() == "no error":  # 요약문장이 factual error를 포함하고 있지 않은 것으로 모델이 예측한 경우
                    pred_labels.append(0)  # no factuality error에 해당하는 label인 0 추가
                else:  # 요약문장이 factual error 포함하고 있는 것으로 모델이 예측한 경우
                    pred_labels.append(1)  # factuality error에 해당하는 label인 1 추가
                pred_types.append(category)  # factuality error 유무와 무관하게 모델이 분류한 category는 전부 pred_types에 추가
            return pred_labels, pred_types
        
        else:  # list 형태로 출력되지 않은 경우 (단일 Json인 경우)
            start_idx = output.find('{')
            end_idx = output.find('}')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)

            pred_labels, pred_types = [], []
            category = output["category"]
            category = category.replace('\n', '').replace('[', '').replace(']', '')
            if category.lower() == "no error":
                pred_labels.append(0)
            else:
                pred_labels.append(1)
            pred_types.append(category)
            return pred_labels, pred_types
        
    except Exception as e:
        
        try:
            subseqs = output.split("category")

            def error_detection(subseq):
                detected = False
                for error_type in ERROR_TYPES:
                    if error_type in subseq:
                        detected = True
                        detected_type = error_type
                if detected:
                    return 1, error_type
                else:
                    return 0, "no error"
                
            pred_labels, pred_types = [], []
            for subseq in subseqs:
                error_label, error_type = error_detection(subseq)
                pred_labels.append(error_label)
                pred_types.append(error_type)
        
            return pred_labels, pred_types
        
        except Exception as e:
            print('parsing error:', e)
            return [], []


'''
Two functions for keyfact alignment
'''
def get_keyfact_alighment_prompt(keyfacts: list, sentences: list):
 
    ''' A function to define the input prompt
    Args:
        keyfacts: the list of keyfacts
        sentences: list of summary sentences
    Return: 
        prompt: the final input prompt
    '''

    summary = [
        '[' 
        + str(line_num + 1) 
        + '] ' 
        + sentence 
        for line_num, sentence in enumerate(sentences)
    ]
    
    summary = '\n'.join(summary)  # list에 담긴 "[LineNumber] Summary Sentence" 형태 str들을 개행문자 이용해 단일 str으로 병합
    num_key_facts = str(len(keyfacts))
    key_facts = '\n'.join(keyfacts)  # list에 담긴 keyfact str들을 개행문자 이용해 단일 str으로 병합
    
    prompt = \
'''
You will receive a summary and a set of key facts for the same transcript. Your task is to assess if each key fact is inferred from the summary.

Instruction:
First, compare each key fact with the summary.
Second, check if the key fact is inferred from the summary and then response "Yes" or "No" for each key fact. If "Yes", specify the line number(s) of the summary sentence(s) relevant to each key fact. 

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "key fact", "response", and "line number":
[{"key fact": "first key fact", "response": "Yes", "line number": [1]}, {"key fact": "second key fact", "response": "No", "line number": []}, {"key fact": "third key fact", "response": "Yes", "line number": [1, 2, 3]}]

Summary:
%s

%s key facts:
%s
''' % (summary, num_key_facts, key_facts)

    return prompt


def parsing_llm_keyfact_alighment_output(output: str):

    ''' A function to parse the output from LLMs based on heuristic rules
    Args:
        output: the output from LLMs
    Return: 
        pred_labels: the binary label for each keyfact (0: no match, 1: match)
        matched_lines: the list of sentence line numbers that align with at least one keyfact
    '''
        
    try:
        # LLM이 KeyFact<->SummarySentences 간의 Alignment를 판단한 출력물을 파싱
        # LLM은 [{"key fact": "first key fact", "response": "yes" 또는 "no", "line number": [1, 5, ...]}, ...] 형태로 출력을 유도받음
        output = output.replace('```', '')
        start_idx = output.find('[')
        output = output[start_idx:]
        output = ast.literal_eval(output)  # str -> python list object 변환

        pred_labels = []  # 각 KeyFact별로 요약문장과의 매치 성공 여부를 binary로 저장
        matched_lines = set()  # KeyFact와의 매치가 하나라도 존재하는 모든 요약문장 넘버들의 집합

        for out in output:  # 각 KeyFact에 대한 (요약 문장들과의) Alignment 여부를 담고 있는 dict 별로 iterate
            category = out["response"]  # KeyFact Alignment 여부
            if category.lower() == "yes":  # 특정 요약 문장(들)과 Align 되는 KeyFact로 판단된 경우
                pred_labels.append(1)  # 해당 KeyFact에 대한 pred_label은 match에 해당하는 1로
            else:  # 어떠한 요약 문장(들)과도 Align 되지 않는 KeyFact로 판단된 경우
                pred_labels.append(0)  # 해당 KeyFact에 대한 pred_label은 no-match에 해당하는 0으로
            
            if 'line number' in out:
                line_nums = out["line number"]  # 해당 KeyFact와 Align되는 요약문장들의 번호 담은 리스트

                for line_num in line_nums:
                    if type(line_num) is str:
                        line_num = line_num.replace('[', '').replace(']', '')  # [1,2,3,...] 형태로 출력되도록 유도했으므로 파싱
                    matched_lines.add(int(line_num))
        
        return pred_labels, list(matched_lines)  # 각 KeyFact들 별로 특정 요약문장(들)과의 Align여부를 담은 바이너리 리스트, 특정 KeyFact(들)과 Align 일어난 문장번호 전체를 담은 리스트
    
    except Exception as e:
        print(e)
        return [], []
    

'''
 Score funtions
'''
def compute_faithfulness_percentage_score(pred_faithfulness_labels):
    faithfulness = 1.0 - sum(pred_faithfulness_labels) / len(pred_faithfulness_labels)  
    return faithfulness

def compute_completeness_percentage_score(pred_alignment_labels):
    completeness = sum(pred_alignment_labels) / len(pred_alignment_labels)  
    return completeness

def compute_conciseness_percentage_score(pred_sentence_line_numbers, num_sentences):
    conciseness = len(pred_sentence_line_numbers) / num_sentences
    return conciseness