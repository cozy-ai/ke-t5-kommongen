import json
import os

file_names = ['kommongen_valid.json', 'kommongen_train.json', 'kommongen_test_1.1.json']

for file_name in file_names:
    concept, scene = [], []
    file_type =  file_name.split('.')[0].split('_')[1]
    if not os.path.exists(f'datasets/final/{file_type}.source') or os.path.exists(f'datasets/final/{file_type}.target'):
        with open(f'datasets/raw/{file_name}', 'r') as f:
            for line in f:
                data = json.loads(line)
                concept.append(f'summarize: {data["concept-set"]}')
                scene.append(data["scene"])
            f.close()
        with open(f'datasets/final/{file_type}.source', 'w') as f:
            f.writelines('\n'.join(concept))
            f.close()
        with open(f'datasets/final/{file_type}.target', 'w') as f:
            f.writelines('\n'.join(scene))
            f.close()