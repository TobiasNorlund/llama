import os
import json
import torch
import sys
from tqdm import tqdm
import numpy as np
from example import load, setup_model_parallel


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_joint_log_probability(logits, input_ids, pad_token_id = 0):
    labels = input_ids[:, 1:].clone().reshape(-1)
    labels[labels == pad_token_id] = -100
    logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    normalized_log_probs = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    normalized_log_probs = normalized_log_probs.reshape(input_ids.shape[0], -1)
    return -normalized_log_probs.sum(-1)


def main(args):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    max_seq_len = 512
    max_batch_size = 32

    llama = load(
        args.model_dir, args.tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    model = llama.model
    tokenizer = llama.tokenizer

    # Load data
    with open(args.pararel_examples) as f:
        examples = [json.loads(line) for line in f]
    with open(args.pararel_options) as f:
        options = [line.strip() for line in f]
    
    # Loop all examples
    log_probs = np.zeros((len(examples), len(options)), dtype=np.float32)
    for i, example in tqdm(enumerate(examples)):
        example_log_probs = []
        for options_batch in batch(options, n=args.batch_size):
            tokens_list = [
                tokenizer.encode(example["question"].replace("<extra_id_0>", option), bos=True, eos=False)
                for option in options_batch
            ]
            tokens = torch.full(
                (len(tokens_list), max(len(ex) for ex in tokens_list)), 
                0
            ).long().cuda()
            for k, t in enumerate(tokens_list):
                tokens[k, : len(t)] = torch.tensor(t).long()

            logits = model.forward(tokens, 0)
            log_prob = get_joint_log_probability(logits, tokens, pad_token_id=0)
            example_log_probs += log_prob.tolist()
        
        log_probs[i, :] = example_log_probs

        example["generation"] = options[np.argmax(example_log_probs)]

    # Save output json with top prediction
    with open(args.output_json, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Save all predictions as numpy array P17.npy
    if args.output_preds is not None:
        np.save(args.output_preds, log_probs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/cephyr/users/tobiasno/Alvis/denitsa-shared/LLaMA/7B"), #required=True)
    parser.add_argument("--tokenizer-path", default="/cephyr/users/tobiasno/Alvis/denitsa-shared/LLaMA/tokenizer.model") #required=True)  
    parser.add_argument("--pararel-examples", default="data/all_n1_atlas/P17.jsonl") #required=True)
    parser.add_argument("--pararel-options", default="data/all_n1_atlas/P17_options.txt") #required=True)
    parser.add_argument("--output-json", default="data/pararel_predictions/7B/P17.jsonl") #required=True)
    parser.add_argument("--output-preds", default="data/pararel_predictions/7B/P17.preds.npy")
    parser.add_argument("--batch-size", default=1, type=int)

    args = parser.parse_args()

    main(args)