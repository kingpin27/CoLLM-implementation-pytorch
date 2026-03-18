# read mtcir.jsonl and output prepped_batch_sz_256.jsonl
import json
import random

MTCIR_FILE_PATH ="./MTCIR/mtcir.jsonl"
OUTPUT_FILE_PATH = "./MTCIR/mtcir_batched.jsonl"
BATCH_SIZE = 2
P = [0.2, 0.4, 0.6, 0.8, 1.0]
TOTAL_EXMAPLES = 17_000_000

with open(MTCIR_FILE_PATH, 'r') as inp_f:
    batch = []
    line_no = 0
    with open(OUTPUT_FILE_PATH, "a", encoding="utf-8") as out_f:
        for line in inp_f:
            example = json.loads(line)
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            random_no = random.random()
            negative = {}
            if random_no <= P[int((min(line_no, TOTAL_EXMAPLES)/TOTAL_EXMAPLES)*len(P))]:
                # add Hard negative corrosponding to example
                negative = {"type": "hard"}
                pass
            else:
                # add soft negative corrosponding to example
                negative = {"type": "soft"}
                pass
            out_f.write(json.dumps(negative, ensure_ascii=False) + "\n")
        line_no += 1