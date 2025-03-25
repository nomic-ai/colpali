from typing import cast

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from colpali_engine.utils.dataset_transformation import load_train_set_colpali_vdr, load_train_set, load_train_set_vdr
from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--lang", type=str, default=None, choices=["it", "en", "fr", "de", "es"])

args = parser.parse_args()

train_sets = load_train_set_vdr(args.lang)['train']
for lang in train_sets:
    print(f"Processing {lang}")
    COMPUTE_EMBEDDINGS = True
    COMPUTE_HARDNEGS = True

    if COMPUTE_HARDNEGS or COMPUTE_EMBEDDINGS:
        print("Loading base model")
        model = BiQwen2_5.from_pretrained(
            "./models/biqwen2_5_1ep_2048_same_source_2e4_vdr_colipali_by_source-best-20250321",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        ).eval()

        print("Loading processor")
        processor = BiQwen2_5_Processor.from_pretrained("./models/biqwen2_5_1ep_2048_same_source_2e4_vdr_colipali_by_source-best-20250321")

    parent_dir = Path(f"data_dir_vdr_{lang}")
    parent_dir.mkdir(exist_ok=True, parents=True)
    if COMPUTE_EMBEDDINGS and not Path(parent_dir / "filtered_dataset_embeddings.pt").exists():
        print("Loading images")
        print("Images loaded")

        document_set = train_sets[lang]
        print("Filtering dataset")
        print(document_set)
        initial_list = document_set["image_filename"]
        _, unique_indices = np.unique(initial_list, return_index=True, axis=0)
        filtered_dataset = document_set.select(unique_indices.tolist())
        filtered_dataset = filtered_dataset.map(
            lambda example: {"image": example["image"], "image_filename": example["image_filename"]}, num_proc=16
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

        ds = torch.stack(ds)

        # save embeddings
        torch.save(ds, parent_dir / "filtered_dataset_embeddings.pt")

    if not COMPUTE_EMBEDDINGS or (parent_dir / "filtered_dataset_embeddings.pt").exists():
        ds = torch.load(parent_dir / "filtered_dataset_embeddings.pt")

    # in case embeddings don't have an exact norm of 1
    ds = ds / torch.norm(ds, dim=1, keepdim=True)

    filtered_dataset = datasets.load_from_disk(parent_dir / "filtered_dataset")
    filenames = list(filtered_dataset["image_filename"])

    margin = 0.95

    if COMPUTE_HARDNEGS:
        # compute hard negatives
        ds = cast(torch.Tensor, ds).to("cuda")

        # iterate on the train set
        mined_hardnegs = []
        negatives = []
        chunk_size = 512
        #
        train_queries = load_dataset("nomic-ai/vdr-multilingula-train", split=lang.split("_")[-1])
        train_queries = train_queries.rename_columns({"id": "image_filename"})
        for i in tqdm(range(0, len(train_queries), chunk_size), desc="Computing hard negatives"):
            samples = train_queries[i : i + chunk_size]
            batch_query = processor.process_queries(samples["query"])
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)

            # compute scores
            scores = torch.einsum("bd,cd->bc", embeddings_query, ds)
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
            neg_files = []
            for idxs in top100:
                neg_files.append([filenames[idx] for idx in idxs])
            negatives.extend(neg_files)

        # save mined hardnegs as txt
        with open(parent_dir / "mined_hardnegs_filtered.txt", "w") as f:
            for item in mined_hardnegs:
                f.write("%s\n" % item)

        with open(parent_dir / "negatives.txt", "w") as f:
            for item in negatives: 
                f.write("%s\n" % item)

    else:
        train_queries = load_dataset("nomic-ai/vdr-multilingula-train", split=lang.split("_")[-1])
        train_queries = train_queries.rename_columns({"id": "image_filename"})
        

    with open(parent_dir / "mined_hardnegs_filtered.txt") as f:
        mined_hardnegs = f.readlines()

    with open(parent_dir / "negatives.txt") as f:
        negatives = f.readlines()


    def mapper_fn(example, idx):
        tmp = {
            # ignore brackets in list written to text file
            "negative_passages": [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")],
            "negatives": [x for x in negatives[idx][1:-2].strip().split(",")],
            "query": example["query"],
            "positive_passages": [filenames.index(example["image_filename"])],
        }

        tmp["gold_in_top_100"] = tmp["positive_passages"][0] in tmp["negative_passages"]
        # remove gold index from negs if it is there
        if tmp["gold_in_top_100"]:
            print("WTF!!!!!!!!!!!!!")
            tmp["negative_passages"].remove(tmp["positive_passages"][0])
        return tmp


    final_dataset = train_queries.map(mapper_fn, with_indices=True, num_proc=16)
    # drop image
    final_dataset = final_dataset.remove_columns("image")
    final_dataset.save_to_disk(parent_dir / "final_dataset")
