import pandas as pd

# As for Web NLG and E2E the datasets are already available as csv. For Abstract Meaning Representation (AMR), the official web page only provides a text file
# ,so we process this file to extract the meaning representations and the target sentences and save the results as csv
def preprocess_amr():
    with open('data/amr/amr-bank-struct-v3.0.txt') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    meaning_representations_not_flattened = list(filter(None, [line if not line.__contains__("#") else [] for line in lines]))
    target_sentences = list(filter(None, [line[8:] if line.__contains__("# ::snt") else [] for line in lines]))
    meaning_representations = []

    for i in range(len(meaning_representations_not_flattened)):
        if meaning_representations_not_flattened[i][0] == "(":
            j = i+1
            while meaning_representations_not_flattened[j][0] != "(":
                j +=1
                if j == len(meaning_representations_not_flattened): break
            meaning_representations.append(''.join(map(str, meaning_representations_not_flattened[i:j])).replace(' ', ''))

    # As for Web NLG and E2E the train/test split is roughly 90/10, so we also use this split for AMR
    pd.DataFrame(list(zip(meaning_representations[:1404], target_sentences[:1404])), columns=['input_text','target_text']).to_csv('data/amr/abstract_meaning_representation_train.csv', index=False)
    pd.DataFrame(list(zip(meaning_representations[1404:], target_sentences[1404:])), columns=['input_text','target_text']).to_csv('data/amr/abstract_meaning_representation_test.csv', index=False)

# Prepend <H>, <R>, <T> (Head, Relation, Tail) Tokens before each triple element in the input column as a second fine tuning technique for Web NLG
def replace_original_triples_with_tokens(data):
  last_token = 'tail'
  for i in range(len(data)):
    split_list = data.iloc[i]['input_text'].split(' | ')
    for j in range(len(split_list)):
      if last_token == 'tail':
        split_list[j] = '<H> ' + split_list[j]
        last_token = 'head'
      elif last_token == 'head':
        split_list[j] = ' <R> ' + split_list[j]
        last_token = 'relation'
      else:
        if last_token == 'relation' and j == len(split_list)-1:
          split_list[j] = ' <T> ' + split_list[j]
          last_token = 'tail'
          data.iloc[i]['input_text'] = ''.join(split_list)
        else:
          split_list[j] = ' <T> ' + split_list[j][:split_list[j].index('&')+2] + ' <H>' + split_list[j][split_list[j].index('&')+2:]
          last_token = 'head'
  return data

# For each equal input text, create a list of all references and finally add them as a new column
def create_references_lists(data):

    # then find the triple which has the highest occurence count in the input_text columns
    # This will be needed to later equalize the lengths of the reference lists for score calculation
    max_occurrence = max(data.groupby(['input_text']).size())
    references_list = []
    inputs_grouped = data.groupby(['input_text'])

    for i in range(len(data)):
        references = list(inputs_grouped.get_group(data.iloc[i]['input_text'])['target_text'])
        for j in range(max_occurrence-len(references)): references.append('')
        references_list.append(references)
    return references_list
    # Now add the references lists as new column to the sorted dataframes
    #references_list_web_nlg = create_references_lists(test_data_web_nlg)
    #test_data_web_nlg['references_list'] = references_list_web_nlg
    #test_data_e2e['references_list'] = create_references_lists(test_data_e2e)
    #test_data_amr['references_list'] = create_references_lists(test_data_amr)

def create_references_files_for_evaluation(data, path):
    inputs_grouped = data.groupby(['input_text'])
    max_occurrence = max(data.groupby(['input_text']).size())
    for i in range(max_occurrence):
        with open(path + 'reference' + str(i), 'w') as file:
            for name, group in inputs_grouped:
                file.write((group.iloc[i][1] + '\n') if len(group) > i else '\n')

    #create_references_files_for_evaluation(test_data_web_nlg, 'data/web_nlg/test/')
    #create_references_files_for_evaluation(test_data_amr, 'data/amr/test/')
