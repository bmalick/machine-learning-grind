import numpy as np
import tiktoken

def pre_tokenize(fname: str, npy_fname: str):
    tokenizer = tiktoken.get_encoding("gpt2")
    pad_id = tokenizer._special_tokens["<|endoftext|>"]

    def tokenize(line):
        tokens = tokenizer.encode(line)
        return tokens

    def buffer_stories(fname):
        buffer = []
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if line=="<|endoftext|>":
                    if buffer:
                        yield " ".join(buffer)
                        buffer = []
                elif line:
                    buffer.append(line)
        if buffer:
            yield " ".join(buffer)

    count = 0
    for story in buffer_stories(fname):
        count += len(tokenize(story)) + 1
    print("Total number of tokens:", count)


    fp = np.memmap(npy_fname, dtype=np.uint16, mode='w+', shape=(count))
    cur = 0
    for story in buffer_stories(fname):
        # print(story)
        # print("="*90)
        tokens = tokenize(story) + [pad_id]
        fp[cur:cur+len(tokens)] = np.array(tokens)
        cur += len(tokens)
    fp.flush()
    print(f"Data saved at {npy_fname}")

if __name__=="__main__":
    pre_tokenize("./TinyStories-valid.txt", "eval-data.dat")
    pre_tokenize("./TinyStories-train.txt", "train-data.dat")
