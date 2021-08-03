import os
import math
import torch
import json

from transformers import AutoTokenizer, AutoModel

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize

def read_file_names(input_data_dir):
    files = os.walk(input_data_dir)

    for path, d, filelist in files:

        filename_list = []
        for name in filelist:
            if name.endswith('.a1') or name.endswith('.a2') or name.endswith('.txt'):
                split_name = name.split('.')
                # print(split_name)
                if len(split_name) == 2:
                    if split_name[0] not in filename_list:
                        filename_list.append(split_name[0])

    # print(filename_list)
    # print(len(filename_list))
    return filename_list

def tokenize_doc(input_dir, rootname):
    # input: a string of rootfile name
    with open(input_dir+rootname+".txt", 'r', encoding="utf-8") as f:
        file_string = f.read()
    
    final_sents, start_idxs = [], []

    sents_wo_enters = file_string.split("\n")

    flag = 0

    for sentence in sents_wo_enters:
        sents = sent_tokenize(sentence)
        for sent in sents:
            final_sents.append(sent)
            start_idx = flag + file_string[flag:].find(sent)
            # start_idx = file_string.find(sent)
            start_idxs.append(start_idx)
            flag = start_idx


    # print(final_sents)
    
    # print(sents)
    # print(start_idxs)
    # for i in range(len(start_idxs)-1):
    #     print(file_string[start_idxs[i]:start_idxs[i]+len(sents[i])])
    #     print(sents[i])
    #     assert (file_string[start_idxs[i]:start_idxs[i]+len(sents[i])]==sents[i])
    return final_sents, start_idxs

def tokenize_sent(sent, tokenizer):
    offsets = tokenizer(sent, return_offsets_mapping=True)["offset_mapping"][1:-1]
    tokens = tokenizer.tokenize(sent)
    # offsets: list of tuples
    return tokens, offsets

def read_entities(input_dir, rootname):
    with open(input_dir+rootname+".a1", 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    entity_dict = {}
    for line in lines:
        splits = line.split("\t")
        ent_name_i, offset_i, text_i = splits[0], splits[1], splits[2]
        ent_type, start, end = offset_i.split()
        if ent_name_i.startswith("T"):
            if ent_name_i not in entity_dict:
                entity_dict.update({ent_name_i: [ent_type, int(start), int(end)]})
    
    return entity_dict

def read_events(input_dir, rootname):
    with open(input_dir+rootname+".a2", 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    trigger_dict, args_dict = {}, {}

    for line in lines:
        # print(line.split("\t"))
        splits = line.split('\t')
        evt_id = splits[0]

        if evt_id.startswith('T'):
            evt_type, start, end = splits[1].split()
            if evt_type != "Entity":
                if evt_id not in trigger_dict:
                    trigger_dict.update({evt_id: [evt_type, int(start), int(end)]})
            
        if evt_id.startswith('E'):
            # print(splits)
            args = splits[1].split()

            if evt_id not in args_dict:
                args_dict.update({evt_id: {"trigger":args[0].split(':')[-1], "args": {}}})
            
            for i,arg in enumerate(args):
                arg_type, end_id = arg.split(':')
                if arg_type.startswith("Theme") or arg_type == "Cause":
                    if arg_type.startswith("Theme"):
                        arg_type = "Theme"
                    args_dict[evt_id]["args"].update({end_id: arg_type})
        
    # print(trigger_dict)
    # for key in args_dict:
    return trigger_dict, args_dict
    # print(trigger_dict)
    # print(args_dict)


def map_to_sent_spans(offsets_list, start, end):
    # offsets_list: [] list of index tuples
    # start, end: idxs
    offset_mappings = offsets_list.copy()
    offset_mappings.insert(0, (-math.inf, -math.inf))
    offset_mappings.append((math.inf, math.inf))

    # first find out the minimum start
    for j in range(1, len(offset_mappings)):
        if offset_mappings[j][0] <= start and offset_mappings[j+1][0] > start:
            break
    span_start = j - 1

    # then find out the minimum end
    for j in range(0, len(offset_mappings)-1):
        if offset_mappings[j][1] < end and offset_mappings[j+1][1] >= end:
            break
    span_end = j + 1

    span_start += 1
    span_end += 1

    return span_start, span_end


def locate_offsets_to_sents(sents, start_idxs, start, end):
    sent_num = len(start_idxs)
    sent_start_idxs = start_idxs.copy()
    found = 0
    for i in range(sent_num):
        sent_range = [sent_start_idxs[i], sent_start_idxs[i]+len(sents[i])]
        if start >= sent_range[0] and end <= sent_range[1]:
            found = 1
            return i

    if not found:
        return -1

def add_bias_to_mappings(mapping_list, start_idx):
    new_mapping_list = []
    for span in mapping_list:
        new_mapping_list.append([span[0]+start_idx, span[1]+start_idx])
    return new_mapping_list

def process_one_doc(input_dir, rootname, tokenizer, complex_list, test=False):
    sentences, start_idxs = tokenize_doc(input_dir, rootname)
    
    ids_list, tokens_list, offsets_list = [], [], []

    for sent in sentences:
        output_i = tokenizer(sent, return_offsets_mapping=True)
        tokens = tokenizer.tokenize(sent)

        ids_list.append(output_i["input_ids"])
        offsets_list.append(output_i["offset_mapping"])

        tokens_list.append(['[CLS]'] + tokens + ['[SEP]'])
    
    output_data_items = []
    for i in range(len(sentences)):
        new_offsets_maps = add_bias_to_mappings(offsets_list[i], start_idxs[i])
        output_data_items.append({"doc_id": rootname, "sent_id": str(i), "sent": sentences[i], "tokens": tokens_list[i], "input_ids": ids_list[i], "doc_pos": new_offsets_maps, "sent_offset": start_idxs[i], "entities": {}, "triggers": {}, "complex_triggers": {},"events": {}})
    # print(output_data_items)
    # read entities and events
    doc_entity_dict = read_entities(input_dir, rootname)
    if not test:
        doc_event_dict, doc_args_dict = read_events(input_dir, rootname)

    # transform each entity and event trigger span in to sentences
    # first for entities:
    # print(doc_entity_dict)

    for entity_id in doc_entity_dict:
        entity_info = doc_entity_dict[entity_id]
        sent_idx = locate_offsets_to_sents(sentences, start_idxs, entity_info[1], entity_info[2])
        if sent_idx != -1:
            start_idx_i = start_idxs[sent_idx]
            
            tokens_start, tokens_end = map_to_sent_spans(offsets_list[sent_idx][1:-1], entity_info[1]-start_idx_i, entity_info[2]-start_idx_i)
            # if sent_idx == 12:
            #     print("sentidx", sent_idx)
            #     print(sentences[sent_idx])
            #     print(offsets_list[sent_idx])
            #     print(tokens_list[sent_idx])
            #     print(entity_info[1]-start_idx_i)
            #     print(entity_info[2]-start_idx_i)
            #     print(sentences[sent_idx][entity_info[1]-start_idx_i: entity_info[2]-start_idx_i])
            #     print(tokens_start)
            #     print(tokens_end)
            #     print(tokens_list[sent_idx][tokens_start: tokens_end])
            output_data_items[sent_idx]["entities"].update({entity_id: {"type": entity_info[0], "span": [tokens_start, tokens_end]}})
        else:
            print("Cross sentence entities")
    
    # for i in range(len(sentences)):
    #     print(i)
    #     print(sentences[i])
    #     print(start_idxs[i])
    #     print('\n')
    # print(doc_event_dict)

    if not test:
        trigger_idx_dict = {}

        for trigger_id in doc_event_dict:
            trigger_info = doc_event_dict[trigger_id]
            sent_idx = locate_offsets_to_sents(sentences, start_idxs, trigger_info[1], trigger_info[2])
            if sent_idx != -1:
                start_idx_i = start_idxs[sent_idx]
                # print(sentences[sent_idx])
                # print(offsets_list[sent_idx])
                # print(tokens_list[sent_idx])
                # print(trigger_info[1]-start_idx_i)
                # print(trigger_info[2]-start_idx_i)
                tokens_start, tokens_end = map_to_sent_spans(offsets_list[sent_idx][1:-1], trigger_info[1]-start_idx_i, trigger_info[2]-start_idx_i)
                # print(tokens_start)
                # print(tokens_end)
                output_data_items[sent_idx]["triggers"].update({trigger_id: {"type": trigger_info[0], "span": [tokens_start, tokens_end]}})
                if trigger_info[0] in complex_list:
                    output_data_items[sent_idx]["complex_triggers"].update({trigger_id: {"type": trigger_info[0], "span": [tokens_start, tokens_end]}})
                    print(trigger_info[0])
            else:
                print("Cross sentence triggers")
            trigger_idx_dict.update({trigger_id: sent_idx})
        
        # finally, the events and arguments

        for event_id in doc_args_dict:
            event_trigger_id = doc_args_dict[event_id]["trigger"]
            event_args = doc_args_dict[event_id]["args"]

            trigger_sent_idx = trigger_idx_dict[event_trigger_id]
            output_data_items[trigger_sent_idx]["events"].update({event_id: {"trigger": event_trigger_id, "args": event_args}})
    
    # for item in output_data_items:
    #     print(item)
    #     print('\n')
    
    # print(trigger_idx_dict)
    return output_data_items


def preprocessing(input_dir, output_dir, tokenizer, complex_list, test=False):
    total_data_items = []
    root_list = read_file_names(input_dir)

    for rootname in root_list:
        data_items = process_one_doc(input_dir, rootname, tokenizer, complex_list, test)
        total_data_items += data_items
    
    with open(output_dir, "w", encoding="utf-8") as f:
        for item in total_data_items:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    # input_dir = "./raw_test/"
    # output_dir = "./test.json"
    input_dir = "./raw_train/"
    output_dir = "./train.json"
    complex_triggers = ["Positive_regulation", "Regulation", "Negative_regulation"]
    # rootname = "PMC-2626671-04-RESULTS_AND_DISCUSSION-03"

    # with open(input_dir+rootname+".txt", 'r', encoding="utf-8") as f:
    #     sss = f.read()
    
    # s, idxs = tokenize_doc(input_dir, rootname)
    # for i,ss in enumerate(s):
    #     print(ss)
    #     print(sss[idxs[i]: idxs[i]+len(ss)])

    #     assert (sss[idxs[i]: idxs[i]+len(ss)] == ss)
    #     print(i)
    
    # print(idxs)

    # print(s)
    # print(idxs)
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    # process_one_doc(input_dir, rootname, t)

    preprocessing(input_dir, output_dir, t, complex_triggers, test=False)
    # preprocessing(input_dir, output_dir, t, complex_triggers, test=True)

    # rootname = "PMID-9018153"
    
    
    # print(s[153:161])

    # # input_dir = "./raw_test"
    # read_file_names(input_dir)

    # # s, idxs = tokenize_doc(input_dir, rootname)
    # # print(s)
    # # print(idxs)

    # # read_entities(input_dir, rootname)
    # # read_events(input_dir, rootname)

    # b = [(0, 3), (4, 14), (15, 22), (22, 25), (26, 34), (34, 36), (37, 39), (39, 40), (40, 42), (42, 45), (45, 46), (47, 53), (54, 56), (57, 60), (61, 68), (68, 69)]
    # c = [(0, 0), (0, 3), (4, 14), (15, 22), (22, 25), (26, 34), (34, 36), (37, 39), (39, 40), (40, 42), (42, 45), (45, 46), (47, 53), (54, 56), (57, 60), (61, 68), (68, 69), (0, 0)]

    # i1 = 26
    # i2 = 59
    # j1, j2 = map_to_sent_spans(b, i1, i2)
    # print(c[j1:j2])

    

    