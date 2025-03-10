from tqdm import tqdm
import random

file_path = 'data/event_pairs.train'
output_file_path = 'data/event_pairs_new.train'
test = False

def split_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        processed_data = []
        positive = []
        negative = []

        for line in tqdm(lines, desc="Processing lines", unit="line"):
            label = process_line(line.strip())
            if label == 1:
                positive.append(line)
            else:
                negative.append(line)

        negative_new = random.sample(negative, 40000)
        processed_data = positive + negative_new

        # shuffle the data
        random.shuffle(processed_data)
        
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for line in processed_data:
            f.write(line)

def process_line(line):
    fields = line.split('\t')
    if test:
        (
            id1, id2, sent1, trg1_s, trg1_e, pp1_1_s, pp1_1_e,
            pp1_2_s, pp1_2_e, time1_s, time1_e, loc1_s, loc1_e,
            sent2, trg2_s, trg2_e, pp2_1_s, pp2_1_e,
            pp2_2_s, pp2_2_e, time2_s, time2_e, loc2_s, loc2_e,
            label
        ) = fields
    else:
        (
            sent1, trg1_s, trg1_e, pp1_1_s, pp1_1_e,
            pp1_2_s, pp1_2_e, time1_s, time1_e, loc1_s, loc1_e,
            sent2, trg2_s, trg2_e, pp2_1_s, pp2_1_e,
            pp2_2_s, pp2_2_e, time2_s, time2_e, loc2_s, loc2_e,
            label
        ) = fields

    label = int(label)
    return label


split_data(file_path)