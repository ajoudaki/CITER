
import json
import sys

if len(sys.argv) < 4:
    print('Usage:', sys.argv[0], 'python agg_arxiv_latex.json <input_metadata_file> <input_file> <output_file>')
    sys.exit(1)

input_metadata_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

metadata = {}
with open(input_metadata_file, 'r') as f:
    for line in f:
        line = json.loads(line)
        arxiv_id = line['id']
        categories = line['categories'].split()
        has_math = any(cat.lower().startswith('math.') for cat in categories)
        if has_math:
            metadata[arxiv_id] = line

print(f'Loaded {len(metadata)} math papers\' metadata')

with open(input_file, 'r') as f, open(output_file, 'w') as f_out:
    cur_arxiv_id = None
    cur_d = None
    num_written = 0
    for i_line, line in enumerate(f):
        if i_line % 10000 == 0:
            print(f'Processing line {i_line//10000}0K, {num_written} written')
        line = json.loads(line)
        path = line['path'].split('/')
        arxiv_id = path[-2]
        if arxiv_id not in metadata:
            continue
        filename = path[-1]
        if not filename.endswith('.tex'):
            continue

        if (cur_arxiv_id is None) or (cur_arxiv_id != arxiv_id):
            if cur_arxiv_id is None:
                cur_arxiv_id = arxiv_id
            else:
                assert cur_d is not None, f'cur_d is None for {cur_arxiv_id}'
                print(json.dumps(cur_d), file=f_out)
                num_written += 1
                cur_arxiv_id = arxiv_id
            cur_d = {
                'arxiv_id': arxiv_id,
                'metadata': metadata[arxiv_id],
                'filenames': [],
                'latex': [],
                'kenlm_perplexity': [],
                'kenlm_sp_perplexity': [],
            }

        cur_d['filenames'].append(filename)
        cur_d['latex'].append(line['text'])
        cur_d['kenlm_perplexity'].append(line['kenlm_perplexity'])
        cur_d['kenlm_sp_perplexity'].append(line['kenlm_sp_perplexity'])