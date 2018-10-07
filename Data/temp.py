train_file_name = 'ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt'
output_file_name = 'first_10000_training_samples.txt'
with open(train_file_name, 'rt', encoding='utf-8') as file:
    result = []
    for line in open(train_file_name, 'rt', encoding='utf-8'):
        result.append(line)
        if (len(result) == 10000):
            break
open(output_file_name, 'wt', encoding='utf-8').write(''.join(result))