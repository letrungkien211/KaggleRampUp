import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import Callback, LambdaCallback
import pickle
import numpy as np
from tqdm import tqdm
import random
from attention import ModelFactory
import argparse
from datetime import datetime
from utils import save_pickle, mkdir
import os

parser = argparse.ArgumentParser(
    description='Train generation model using original attention model.')

parser.add_argument('--input_file', required=True,
                    help='Input file. Format: Source TAB TARGET')
parser.add_argument('--vocab_size', required=False,
                    default=30000, help='Vocabulary sizes')
parser.add_argument('--num_samples', required=False, type=int,
                    default=sys.maxsize, help='Number of samples to read')
parser.add_argument('--num_samples_tokenizer', required=False,
                    default=1000000, help='Number of samples to create tokenizer')
parser.add_argument('--embedding_dim_decoder', type=int,
                    required=False, default=256, help='Embedding dim for decoder')
parser.add_argument('--embedding_dim_encoder', type=int,
                    required=False, default=256, help='Embedding dim for encoder')
parser.add_argument('--hidden_dim_encoder', type=int,
                    required=False, default=256, help='Embedding dim for decoder')
parser.add_argument('--hidden_dim_decoder', type=int,
                    required=False, default=256, help='Embedding dim for decoder')
parser.add_argument('--batch_size', type=int, required=False,
                    default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=1,
                    required=False, help='Number of epochs')
parser.add_argument('--input_format', type=str, default='cqa_triplets',
                    help='Input format: qa_pairs or cqa_triplets')
parser.add_argument('--initial_epoch', type=int, default=0, required=False)

parser.add_argument('--root_dir', type=str, required=False, default='../data/attention_join/')
parser.add_argument('--max_input_char_len', type=int, default=150)
parser.add_argument('--max_output_char_len', type=int, default=50)

args = parser. parse_args()

args.logs_dir = os.path.join(args.root_dir, 'logs/')
args.models_dir = os.path.join(args.root_dir, 'models/')

mkdir(args.logs_dir)
mkdir(args.models_dir)


mkdir(args.root_dir)
# Parameters

# Pre process
cnt = 0
input_texts = []  # source sentence
target_texts = []  # target sentence
# target sentence offset by 1print('# lines loaded: {0}'.format(len(input_texts)))
target_texts_inputs = []

print('Read input file')
with open(args.input_file, encoding='utf-8') as f:
    for line in tqdm(f):
        if(cnt >= args.num_samples):
            break
        splits = [y.strip() for y in line.split('\t')]

        if args.input_format == 'cqa_triplets':
            if(len(splits) != 3):
                continue
            if not splits[1] or not splits[2]:
                continue
            input_text = splits[0] + ' <cq> ' + splits[1]
            target_text = splits[2] + ' <eos>'
            target_text_input = '<sos> ' + splits[2]  # offset by 1
        else:
            if(len(splits) != 2):
                continue
            input_text = splits[0] + ' <cq> ' + splits[1]
            target_text = splits[1] + ' <eos>'
            target_text_input = '<sos> ' + splits[1]  # offset by 1

        if(len(input_text) > args.max_input_char_len or len(target_text) > args.max_output_char_len):
            continue

        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)
        cnt += 1

print('Total lines read: ', len(input_texts))

print('Fit tokenizers')
# Tokenizer
tokenizer_inputs = Tokenizer(num_words=args.vocab_size, filters='')
tokenizer_outputs = Tokenizer(num_words=args.vocab_size, filters='')

if len(input_texts) > args.num_samples_tokenizer:
    tokenizer_inputs.fit_on_texts(random.sample(
        input_texts, args.num_samples_tokenizer))
    tokenizer_outputs.fit_on_texts(random.sample(
        [x + ' <eos>' for x in target_texts_inputs], args.num_samples_tokenizer))
else:
    tokenizer_inputs.fit_on_texts(input_texts)
    tokenizer_outputs.fit_on_texts([x + ' <eos>' for x in target_texts_inputs])

print('Convert texts to sequences')
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(
    target_texts_inputs)
target_squences = tokenizer_outputs.texts_to_sequences(target_texts)

# Calculate statistics
num_words_output = min(args.vocab_size,  len(tokenizer_outputs.word_index)) + 1
num_words_input = min(args.vocab_size, len(tokenizer_inputs.word_index)) + 1

save_pickle(os.path.join(args.root_dir, 'tokenizer_inputs_{0}'.format(num_words_input)), tokenizer_inputs)
save_pickle(os.path.join(args.root_dir, 'tokenizer_inputs_{0}'.format(num_words_output)), tokenizer_outputs)

max_len_input = max(len(s) for s in input_sequences)
max_len_target = max(len(s) for s in target_squences)

# Padding
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
decoder_inputs = pad_sequences(
    target_sequences_inputs, maxlen=max_len_target, padding='post')
decoder_targets = pad_sequences(
    target_squences, maxlen=max_len_target, padding='post')

print(num_words_input, num_words_output, max_len_input, max_len_target)


class SequenceGenerator(Sequence):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets, batch_size,  max_len_target, hidden_dim_decoder, num_words_output):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

        self.batch_size = batch_size
        self.max_len_target = max_len_target
        self.hidden_dim_decoder = hidden_dim_decoder
        self.num_words_output = num_words_output

    def __len__(self):
        return len(self.encoder_inputs) // self.batch_size + (len(self.encoder_inputs) % self.batch_size != 0)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = min((idx+1)*self.batch_size, len(self.encoder_inputs))

        encoder_inputs = self.encoder_inputs[start:end]
        decoder_inputs = self.decoder_inputs[start:end]
        decoder_targets = self.decoder_targets[start:end]

        z = np.zeros((len(encoder_inputs), self.hidden_dim_decoder))

        decoder_targets_one_hot = np.zeros(
            (
                len(encoder_inputs),
                self.max_len_target,
                self.num_words_output
            ),
            dtype='float32'
        )

        for i, d in enumerate(decoder_targets):
            for t, word in enumerate(d):
                decoder_targets_one_hot[i, t, word] = 1

        return ([encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot)

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

# model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

num_train = int(len(encoder_inputs) * 0.9)

train_generator = SequenceGenerator(encoder_inputs[0:num_train], decoder_inputs[0:num_train],
                                    decoder_targets[0:num_train], args.batch_size, max_len_target, args.hidden_dim_decoder, num_words_output)
validation_generator = SequenceGenerator(encoder_inputs[num_train:], decoder_inputs[num_train:],
                                         decoder_targets[num_train:], args.batch_size, max_len_target, args.hidden_dim_decoder, num_words_output)

print('Start training')

callbacks = [TensorBoard(os.path.join(args.logs_dir, 'attention-{0}'.format(datetime.now().isoformat().replace(':','-').split('.')[0]))),
             ModelCheckpoint(os.path.join(args.models_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5'), save_best_only=True)]

r = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator),
    epochs=args.epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    initial_epoch=args.initial_epoch
    )

for i in range(min(20, len(encoder_inputs))):
  output = inference_model.predict(np.array(encoder_inputs[i:i+1]), tokenizer_outputs.word_index['<sos>'], tokenizer_outputs.word_index['<eos>'])
  output_sentences = tokenizer_outputs.sequences_to_texts([list(output)])
  print(input_texts[i], '<qa>', output_sentences[0])
