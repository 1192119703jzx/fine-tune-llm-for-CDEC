from tqdm import tqdm

# Uncomment the following line to use the SRL extractor
#from SRL_extractor import extract_srl

class EventPairDataset:
    def __init__(self, file_path, srl=False, test=False):
        self.path = file_path
        self.test = test
        self.srl = srl
        self.data = self.load()

    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            processed_data = []

            for line in tqdm(lines, desc="Processing lines", unit="line"):
                processed = self.process_line(line.strip())
                if processed:
                  processed_data.append(processed)

        return processed_data
            
    def process_line(self, line):
        if not line.strip():  # Skip empty lines
          return None

        fields = line.split('\t')
        if self.test:
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

        '''
        # Uncomment the following lines to use the SRL extractor
        if self.srl:
            pp1_1_s, pp1_1_e, pp1_2_s, pp1_2_e, time1_s, time1_e, loc1_s, loc1_e = extract_srl(sent1, ' '.join(sent1.split()[int(trg1_s): int(trg1_e)+1]))
            pp2_1_s, pp2_1_e, pp2_2_s, pp2_2_e, time2_s, time2_e, loc2_s, loc2_e = extract_srl(sent2, ' '.join(sent2.split()[int(trg2_s): int(trg2_e)+1]))
        '''

        sentence1 = self.process_tags(sent1, trg1_s, trg1_e, pp1_1_s, pp1_1_e,
            pp1_2_s, pp1_2_e, time1_s, time1_e, loc1_s, loc1_e)
        sentence2 = self.process_tags(sent2, trg2_s, trg2_e, pp2_1_s, pp2_1_e,
            pp2_2_s, pp2_2_e, time2_s, time2_e, loc2_s, loc2_e)
        
        return {
            "sentence1": sentence1, "sentence2": sentence2,
            "label": int(label)
        }
    
    def process_tags(self, sentence, trg_s, trg_e, pp1_s, pp1_e, pp2_s, pp2_e, time_s, time_e, loc_s, loc_e):
        tokens = [[word] for word in sentence.split()]
        #print(tokens)
        trg_s, trg_e, pp1_s, pp1_e, pp2_s, pp2_e, time_s, time_e, loc_s, loc_e = map(int, [trg_s, trg_e, pp1_s, pp1_e, pp2_s, pp2_e, time_s, time_e, loc_s, loc_e])
        if trg_s != -1 and trg_e != -1:
            tokens[trg_s].insert(0, "<TRG>")
            tokens[trg_e].append("</TRG>")

        if pp1_s != -1 and pp1_e != -1:
            tokens[pp1_s].insert(0, "<ARG>")
            tokens[pp1_e].append("</ARG>")

        if pp2_s != -1 and pp2_e != -1:
            tokens[pp2_s].insert(0, "<ARG>")
            tokens[pp2_e].append("</ARG>")

        if time_s != -1 and time_e != -1:
            tokens[time_s].insert(0, "<TIME>")
            tokens[time_e].append("</TIME>")

        if loc_s != -1 and loc_e != -1:
            tokens[loc_s].insert(0, "<LOC>")
            tokens[loc_e].append("</LOC>")

        tokens = [word for sublist in tokens for word in sublist]
        return " ".join(tokens)

