
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

with open(args.input_file, "r") as f, open(args.output_file, "w") as fout:
    for i_line, line in enumerate(f):
        if i_line % 10000 == 0:
            print(f"Processing line {i_line//10000}0k", file=sys.stderr)
        line = json.loads(line)
        lemmas = line["lemmas"]
        theorems = line["theorems"]
        for i_lemma, lemma in enumerate(lemmas):
            idx = f"{i_line}-lemma-{i_lemma}"
            fout.write(json.dumps({
                "id": idx,
                "text": lemma,
            }) + "\n")
        for i_theorem, theorem in enumerate(theorems):
            idx = f"{i_line}-theorem-{i_theorem}"
            fout.write(json.dumps({
                "id": idx,
                "text": theorem,
            }) + "\n")