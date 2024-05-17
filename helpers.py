import numpy as np
import pandas as pd

def tokenize_and_label_7label(input,input_with_annotations,tokenizer,use_overflow=False):

    token_array = np.zeros((len(input)),dtype=object)
    

    char_dict_mapping = {}
    char_array_orig = []
    for i,note_id in enumerate(input.index):
        char_dict_mapping[note_id] = i
        char_array_orig.append(np.zeros(len(input.loc[note_id,'text'])))

    for index,row in input_with_annotations.iterrows():
        if row['snomed_base'] == 'procedure':
            char_array_orig[char_dict_mapping[index]][row['start']] = 1
            char_array_orig[char_dict_mapping[index]][row['start']+1:row['end']] = 2
        if row['snomed_base'] == 'finding':
            char_array_orig[char_dict_mapping[index]][row['start']] = 3
            char_array_orig[char_dict_mapping[index]][row['start']+1:row['end']] = 4
        if row['snomed_base'] == 'body structure':
            char_array_orig[char_dict_mapping[index]][row['start']] = 5
            char_array_orig[char_dict_mapping[index]][row['start']+1:row['end']] = 6
    
    if use_overflow:
        encoding = tokenizer(list(input['text'].values), padding='max_length',
                        truncation=True, return_offsets_mapping=True)
    else:
        encoding = tokenizer(list(input['text'].values),
                    truncation=False, return_offsets_mapping=True)

    map_token_to_char = encoding['offset_mapping']
    for i in range(len(map_token_to_char)):
        token_array[i] = np.zeros(len(encoding['input_ids'][i]),dtype=np.int32)
        for k in range(len(token_array[i])):
            token_array[i][k] = 0
            #Finds the first non-zero in the token and sets that value for the token
            for val in char_array_orig[i][ map_token_to_char[i][k][0] : map_token_to_char[i][k][1] ]:
                if val > 0:
                    token_array[i][k] = val
                    break
                    
#     del(encoding['offset_mapping'])
    input_tokens = encoding

    input_tokens['labels'] = []
    for i in range(len(input)):
        input_tokens['labels'].append(token_array[i])
    
    train_df = pd.DataFrame(input_tokens.__dict__['data'])
    note_indexes = list(input.index)
    #Add Overflow
    if use_overflow:
        for t in range(len(input_tokens['input_ids'])):
            for o in encoding[t].overflowing:

                overflow_tokens = np.zeros(len(o.ids),dtype=np.int32)
                # print(len(overflow_tokens))

                for k in range(len(overflow_tokens)):
                    # token_array[i][k] = char_array_orig[i][ map_token_to_char[i][k][0] ]
                    overflow_tokens[k] = 0
                    for val in char_array_orig[t][ o.offsets[k][0] : o.offsets[k][1] ]:
                        if val > 0:
                            overflow_tokens[k] = val
                            break

                index = len(train_df)
                note_indexes.append(list(input.index)[t])
                train_df.loc[index,'input_ids'] = o.ids
                train_df.loc[index,'offset_mapping'] = o.offsets
                if 'token_type_ids' in list(train_df.columns):
                    train_df.loc[index,'token_type_ids'] = o.type_ids
                train_df.loc[index,'attention_mask'] = o.attention_mask
                train_df.loc[index,'labels'] = overflow_tokens

    train_df['note_id'] = note_indexes
    train_df = train_df.set_index('note_id')

    return train_df, token_array, map_token_to_char, char_array_orig

def tokenize_and_label_3label(input,input_with_annotations,tokenizer,use_overflow=False):

    token_array = np.zeros((len(input)),dtype=object)
    

    char_dict_mapping = {}
    char_array_orig = []
    for i,note_id in enumerate(input.index):
        char_dict_mapping[note_id] = i
        char_array_orig.append(np.zeros(len(input.loc[note_id,'text'])))

    for index,row in input_with_annotations.iterrows():
        char_array_orig[char_dict_mapping[index]][row['start']] = 1
        char_array_orig[char_dict_mapping[index]][row['start']+1:row['end']] = 2

    if use_overflow:
        encoding = tokenizer(list(input['text'].values), padding='max_length',
                        truncation=True, return_offsets_mapping=True)
    else:
        encoding = tokenizer(list(input['text'].values),
                    truncation=False, return_offsets_mapping=True)

    map_token_to_char = encoding['offset_mapping']
    for i in range(len(map_token_to_char)):
        token_array[i] = np.zeros(len(encoding['input_ids'][i]),dtype=np.int32)
        for k in range(len(token_array[i])):
            token_array[i][k] = 0
            #Finds the first non-zero in the token and sets that value for the token
            for val in char_array_orig[i][ map_token_to_char[i][k][0] : map_token_to_char[i][k][1] ]:
                if val > 0:
                    token_array[i][k] = val
                    break
        
    # del(encoding['offset_mapping'])
    input_tokens = encoding

    input_tokens['labels'] = []
    for i in range(len(input)):
        input_tokens['labels'].append(token_array[i])

    train_df = pd.DataFrame(input_tokens.__dict__['data'])
    note_indexes = list(input.index)
    #Add Overflow
    if use_overflow:
        for t in range(len(input_tokens['input_ids'])):
            for o in encoding[t].overflowing:

                overflow_tokens = np.zeros(len(o.ids),dtype=np.int32)
                # print(len(overflow_tokens))

                for k in range(len(overflow_tokens)):
                    # token_array[i][k] = char_array_orig[i][ map_token_to_char[i][k][0] ]
                    overflow_tokens[k] = 0
                    for val in char_array_orig[t][ o.offsets[k][0] : o.offsets[k][1] ]:
                        if val > 0:
                            overflow_tokens[k] = val
                            break

                index = len(train_df)
                note_indexes.append(list(input.index)[t])
                train_df.loc[index,'input_ids'] = o.ids
                train_df.loc[index,'offset_mapping'] = o.offsets
                if 'token_type_ids' in list(train_df.columns):
                    train_df.loc[index,'token_type_ids'] = o.type_ids
                train_df.loc[index,'attention_mask'] = o.attention_mask
                train_df.loc[index,'labels'] = overflow_tokens

    train_df['note_id'] = note_indexes
    train_df = train_df.set_index('note_id')

    return train_df, token_array, map_token_to_char, char_array_orig