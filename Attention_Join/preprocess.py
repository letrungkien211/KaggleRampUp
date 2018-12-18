import sys

## Pre process data
class PreProcess():
    def __init__(self):
        pass

    def prep(self, input_file, num_samples = sys.maxsize, ):
        cnt = 0;
        input_texts = [] # source sentence
        target_texts = [] # target sentence
        target_texts_inputs = [] # target sentence offset by 1
        with open(input_file, encoding='utf-8') as f:
            for line in f:
                if(cnt > num_samples):
                    break;
                if(cnt%1000==0):
                    print('# lines loaded: {0}'.format(cnt))
                splits =[y for y in [x.strip() for x in line.strip().split('\t')] if y]
                if(len(splits)!=2):
                    continue
                input_text = splits[0]
                target_text = splits[1] + ' <eos>'  
                target_text_input = '<sos> ' + splits[1] # offset by 1

                input_texts.append(input_text)
                target_texts.append(target_text)
                target_texts_inputs.append(target_text_input)
                cnt+=1
        print('# lines loaded: {0}'.format(len(input_texts)))

        