from typing import cast

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.utils.dataset_transformation import load_train_set_colpali_vdr, load_train_set
from pathlib import Path

train_set = load_train_set()


COMPUTE_EMBEDDINGS = True
COMPUTE_HARDNEGS = True

if COMPUTE_HARDNEGS or COMPUTE_EMBEDDINGS:
    print("Loading base model")
    model = ColQwen2_5.from_pretrained(
        "./models/colqwen2_5_train_single_source_3ep_r32_512bs_vdr",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    ).eval()

    print("Loading processor")
    processor = ColQwen2_5_Processor.from_pretrained("./models/colqwen2_5_train_single_source_3ep_r32_512bs_vdr")

parent_dir = Path("data_dir")
parent_dir.mkdir(exist_ok=True, parents=True)
if COMPUTE_EMBEDDINGS and not Path(parent_dir / "filtered_dataset_embeddings.pt").exists():
    print("Loading images")
    print("Images loaded")

    document_set = train_set["train"]
    print("Filtering dataset")
    print(document_set)
    initial_list = document_set["image_filename"]
    _, unique_indices = np.unique(initial_list, return_index=True, axis=0)
    filtered_dataset = document_set.select(unique_indices.tolist())
    filtered_dataset = filtered_dataset.map(
        lambda example: {"image": example["image"], "image_filename": example["image_filename"]}, num_proc=32
    )
    # keep only column image and image_filename and source if it exists
    cols_to_remove = [col for col in filtered_dataset.column_names if col not in ["image", "image_filename"]]
    filtered_dataset = filtered_dataset.remove_columns(cols_to_remove)
    # save it
    print("Saving filtered dataset")
    print(filtered_dataset)
    if not Path(parent_dir / "filtered_dataset").exists():
        filtered_dataset.save_to_disk(parent_dir / "filtered_dataset", max_shard_size="200MB")

    print("Processing images")
    # run inference - docs
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda x: processor.process_images([a["image"] for a in x]),
    )
    print("Computing embeddings")

    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # can't save because tensors are different shapes due to variable image token length and no pooling
    # ds = torch.stack(ds)

    # save embeddings
    torch.save(ds, parent_dir / "filtered_dataset_embeddings.pt")

if not COMPUTE_EMBEDDINGS or (parent_dir / "filtered_dataset_embeddings.pt").exists():
    ds = torch.load(parent_dir / "filtered_dataset_embeddings.pt")

ds = [a / torch.norm(a) for a in ds]

filtered_dataset = datasets.load_from_disk(parent_dir / "filtered_dataset")
filenames = list(filtered_dataset["image_filename"])

margin = 0.95

if COMPUTE_HARDNEGS:
    # compute hard negatives
    ds = cast(torch.Tensor, ds).to("cuda")

    # iterate on the train set
    mined_hardnegs = []
    chunk_size = 512
    for i in tqdm(range(0, len(train_set["train"]), chunk_size), desc="Computing hard negatives"):
        samples = train_set["train"][i : i + chunk_size]
        batch_query = processor.process_queries(samples["query"])
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)

        # compute scores
        all_scores = []
        for i in tqdm(range(0, len(ds), 128)):
            # all documents in batch will have same number of tokens, compute scores in subbatches across documents
            curr_ds = ds[i : i + 128]
            curr_score = torch.einsum("bnd,csd->bcns", embeddings_query, curr_ds).max(dim=3)[0].sum(dim=2)
            all_scores.append(curr_score)

        scores = torch.cat(all_scores, dim=0)
        pos_idxs = [filenames.index(example) for example in samples['image_filename']]
        pos_scores = scores[torch.arange(len(pos_idxs)), pos_idxs]

        # Create mask for scores less than positive score * margin
        pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, scores.size(1))
        valid_scores_mask = scores < (pos_scores_expanded * margin)
        # Set invalid scores to -inf so they won't be selected
        masked_scores = scores.clone()
        masked_scores[~valid_scores_mask] = float('-inf')
        
        # Get top 100 indexes from valid scores
        top100 = masked_scores.topk(min(100, valid_scores_mask.sum(dim=1).max().item()), dim=1).indices
        # indices to list
        top100 = top100.tolist()
        # append to mined_hardnegs
        mined_hardnegs.extend(top100)

    # save mined hardnegs as txt
    with open(parent_dir / "mined_hardnegs_filtered.txt", "w") as f:
        for item in mined_hardnegs:
            f.write("%s\n" % item)


with open(parent_dir / "mined_hardnegs_filtered.txt") as f:
    mined_hardnegs = f.readlines()


def mapper_fn(example, idx):
    tmp = {
        "negative_passages": [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")],
        "query": example["query"],
        "positive_passages": [filenames.index(example["image_filename"])],
    }

    tmp["gold_in_top_100"] = tmp["positive_passages"][0] in tmp["negative_passages"]
    # remove gold index from negs if it is there
    if tmp["gold_in_top_100"]:
        print("WTF!!!!!!!!!!!!!")
        tmp["negative_passages"].remove(tmp["positive_passages"][0])
    return tmp


final_dataset = train_set["train"].map(mapper_fn, with_indices=True, num_proc=16)
# drop image
final_dataset = final_dataset.remove_columns("image")
final_dataset.save_to_disk(parent_dir / "final_dataset")
