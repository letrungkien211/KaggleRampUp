import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
import pickle
import numpy as np
from tqdm import tqdm

from attention import ModelFactory


import argparse

parser = argparse.ArgumentParser(description='Train generation model using original attention model.')

parser.add_argument('--input_file', required=True, help='Input file. Format: Source TAB TARGET')
parser.add_argument('--vocab_size', required=False, default=30000, help='Vocabulary sizes')
parser.add_argument('--num_samples', required=False, type=int, default=sys.maxsize, help='Number of samples to read')
parser.add_argument('--num_samples_tokenizer', required=False, default=sys.maxsize, help='Number of samples to create tokenizer')
parser.add_argument('--embedding_dim_decoder', type=int, required=False, default=256, help='Embedding dim for decoder')
parser.add_argument('--embedding_dim_encoder', type=int, required=False, default=256, help='Embedding dim for encoder')
parser.add_argument('--hidden_dim_encoder', type=int, required=False, default=1024, help='Embedding dim for decoder')
parser.add_argument('--hidden_dim_decoder', type=int, required=False, default=1024, help='Embedding dim for decoder')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=1, required=False, help='Number of epochs')
parser.add_argument('--input_format', type=str, default='qa_pairs', help='Input format: qa_pairs or cqa_triplets')

args = parser.parse_args()

## Parameters

## Pre process
cnt = 0
input_texts = [] # source sentence
target_texts = [] # target sentence
target_texts_inputs = [] # target sentence offset by 1print('# lines loaded: {0}'.format(len(input_texts)))


with open(args.input_file, encoding='utf-8') as f:
    for line in tqdm(f):
        if(cnt >= args.num_samples):
            break
        splits =[y for y in [x.strip() for x in line.strip().split('\t')] if y]

        if args.input_format == 'cqa_triplets':

            if(len(splits)!=3):
                continue
            input_text = splits[0] + ' <cq_separator> ' + splits[1]
            target_text = splits[2] + ' <eos>'  
            target_text_input = '<sos> ' + splits[2] # offset by 1
        else:
            splits =[y for y in [x.strip() for x in line.strip().split('\t')] if y]
            if(len(splits)!=2):
                continue
            input_text = splits[0] + ' <cq_separator> ' + splits[1]
            target_text = splits[1] + ' <eos>'  
            target_text_input = '<sos> ' + splits[1] # offset by 1

        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)
        cnt+=1

## Tokenizer 
tokenizer_inputs = Tokenizer(num_words=args.vocab_size, filters='')
tokenizer_inputs.fit_on_texts(input_texts)

tokenizer_outputs = Tokenizer(num_words=args.vocab_size, filters='')
tokenizer_outputs.fit_on_texts([x + ' <eos>' for x in target_texts_inputs])

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
target_squences = tokenizer_outputs.texts_to_sequences(target_texts)

num_words_output = len(tokenizer_outputs.word_index) + 1
num_words_input = len(tokenizer_inputs.word_index) + 1

# Calculate statistics
max_len_input = max(len(s) for s in input_sequences)
max_len_target = max(len(s) for s in target_squences)

# Padding
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
decoder_targets = pad_sequences(target_squences, maxlen= max_len_target, padding='post')
decoder_targets_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_target,
        num_words_output
    ),
    dtype='float32'
)

for i, d in tqdm(enumerate(decoder_targets)):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1


class SequenceGenerator(Sequence):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass

## Create model

model, inference_model = ModelFactory.create(
    max_len_input,
    max_len_target,
    args.embedding_dim_encoder,
    args.embedding_dim_decoder,
    args.hidden_dim_encoder,
    args.hidden_dim_decoder,
    num_words_input,
    num_words_output
)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

z = np.zeros((len(input_texts), args.hidden_dim_decoder))    

r = model.fit(
  [encoder_inputs, decoder_inputs, z, z], 
  decoder_targets_one_hot,
  batch_size=args.batch_size,
  epochs=args.epochs,
  validation_split=0.2
)

for i in range(min(20, len(encoder_inputs))):
  output = inference_model.predict(np.array(encoder_inputs[i:i+1]), tokenizer_outputs.word_index['<sos>'], tokenizer_outputs.word_index['<eos>'])
  output_sentences = tokenizer_outputs.sequences_to_texts([list(output)])
  print(output)
  print(input_texts[i], '<qa_separator>', output_sentences[0])