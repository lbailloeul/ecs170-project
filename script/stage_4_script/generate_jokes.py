from pathlib import Path
import argparse
import random
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN


def load_data():
    data_obj = Dataset_Loader('stage 4 text generation', '')
    data_obj.dataset_source_folder_path = str(REPO_ROOT / 'data' / 'stage_4_data' / 'text_generation') + '/'
    data_obj.dataset_file_name = 'data'
    data_obj.sequence_length = 32
    data_obj.train_ratio = 0.8
    data_obj.min_freq = 2
    data_obj.max_vocab_size = 3000
    return data_obj.load()


def load_model(data, checkpoint_path):
    model = Method_RNN(
        'recurrent neural network text generator',
        '',
        embed_size=128,
        hidden_size=256,
        rnn_type='lstm',
        dropout=0.4,
        max_epoch=40,
        batch_size=128,
        learning_rate=0.0005,
        weight_decay=0.0001,
        early_stopping_patience=6,
    )
    model.data = data
    model._configure_from_data()
    state = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(state)
    return model


def random_prompt(model, data, rng):
    joke = rng.choice(data['train_jokes'])
    tokens = model._tokenize(joke)[:3]
    return model._detokenize(tokens) or 'why did the'


def main():
    parser = argparse.ArgumentParser(description='Generate jokes from the trained stage 4 RNN model.')
    parser.add_argument('--prompt', default='why did the', help='Seed text to start generation.')
    parser.add_argument('--random-prompt', action='store_true', help='Use a random 3-token prompt from the training jokes.')
    parser.add_argument('--n', type=int, default=5, help='Number of jokes to generate.')
    parser.add_argument('--max-new-tokens', type=int, default=28, help='Maximum generated tokens per joke.')
    parser.add_argument('--temperature', type=float, default=0.55, help='Sampling temperature.')
    parser.add_argument('--top-k', type=int, default=30, help='Top-k sampling cutoff.')
    parser.add_argument('--top-p', type=float, default=0.85, help='Nucleus sampling cutoff.')
    parser.add_argument('--repetition-penalty', type=float, default=1.15, help='Penalty for recently repeated tokens.')
    parser.add_argument('--seed', type=int, default=1200, help='Random seed for reproducible sampling.')
    parser.add_argument(
        '--checkpoint',
        default=str(REPO_ROOT / 'result' / 'stage_4_result' / 'Stage_4_model.pt'),
        help='Path to the saved model checkpoint.',
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = load_data()
    model = load_model(data, args.checkpoint)

    prompt = args.prompt
    if args.random_prompt:
        prompt = random_prompt(model, data, rng)
        print('Prompt:', prompt)

    for index in range(args.n):
        torch.manual_seed(args.seed + index)
        joke = model.generate(
            seed_text=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f'{index + 1}. {joke}')


if __name__ == '__main__':
    main()
