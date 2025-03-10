import torch
from allennlp.predictors.predictor import Predictor

device = 0 if torch.cuda.is_available() else -1

# Load SRL predictor
srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
    cuda_device=device
)

def extract_srl(sentence, trigger):
    # get predictions
    result = srl_predictor.predict(sentence=sentence)
    events = {}

    for item in result["verbs"]:
        v = item["verb"]
        description = item["description"]
        participant1 = None
        participant2 = None
        time = None
        location = None

        if "ARG0:" in description:
                participant1 = description.split("ARG0:")[1].split("]")[0].strip()

        if "ARG1:" in description:
                participant2 = description.split("ARG1:")[1].split("]")[0].strip() if "ARG1:" in description else None

        if "ARGM-TMP:" in description:
                time = description.split("ARGM-TMP:")[1].split("]")[0].strip() if "ARGM-TMP:" in description else None

        if "ARGM-LOC:" in description:
                location = description.split("ARGM-LOC:")[1].split("]")[0].strip() if "ARGM-LOC:" in description else None

        events[v] = {
            "participant1": participant1,
            "participant2": participant2,
            "time": time,
            "location": location
        }

    output = None
    for verb in events:
        if trigger in verb:
            output = events[verb]
            break
    if output is None:
        if events:
            output = next(iter(events.values()))
        else:
            return -1, -1, -1, -1, -1, -1, -1, -1

    pp1_1_s, pp1_1_e = find_index(sentence, output["participant1"])
    pp1_2_s, pp1_2_e = find_index(sentence, output["participant2"])
    time1_s, time1_e = find_index(sentence, output["time"])
    loc1_s, loc1_e = find_index(sentence, output["location"])

    return pp1_1_s, pp1_1_e, pp1_2_s, pp1_2_e, time1_s, time1_e, loc1_s, loc1_e
    
def find_index(sentence, phrase):
    if phrase is None:
        return -1, -1
    else:
        sentence = sentence.split()
        phrase = phrase.split()
        for i in range(len(sentence)):
            if sentence[i:i+len(phrase)] == phrase:
                return i, i + len(phrase) - 1
        return -1, -1