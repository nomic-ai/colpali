import os
from typing import List, Tuple, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "0") == "1"


def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)


def load_train_set() -> DatasetDict:
    ds_path = "colpali_train_set"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path))
    return ds_dict

    
def load_train_set_colpali_vdr() -> DatasetDict:
    ds_paths = [
        ("nomic-ai/colpali_train_set_split_by_source", None),
        ("nomic-ai/vdr-multilingula-train", "it"),
        ("nomic-ai/vdr-multilingula-train", "en"),
        ("nomic-ai/vdr-multilingula-train", "fr"),
        ("nomic-ai/vdr-multilingula-train", "de"),
        ("nomic-ai/vdr-multilingula-train", "es"),
    ]
    ds_tot = {}
    
    for (path, split) in ds_paths:
        if split is None:
            ds = cast(DatasetDict, load_dataset(path, num_proc=4))
            ds_tot = {**ds_tot, **ds}
        else:
            ds = cast(Dataset, load_dataset(path, split=split, num_proc=4))
            ds_tot[f'{path.split("/")[1]}_{split}'] = ds
            
    dataset = cast(DatasetDict, DatasetDict(ds_tot))
    
    ds_dict = DatasetDict({"train": dataset, "test": None})
    return ds_dict

def load_train_set_ir_negs_vdr() -> Tuple[DatasetDict, Dataset, str]:
    ds_paths = [
        ("nomic-ai/colpali-queries-mined-20250321-by-source", None),
        ("nomic-ai/vdr-multilingual-train-hn-mine", "it"),
        ("nomic-ai/vdr-multilingual-train-hn-mine", "en"),
        ("nomic-ai/vdr-multilingual-train-hn-mine", "fr"),
        ("nomic-ai/vdr-multilingual-train-hn-mine", "de"),
        ("nomic-ai/vdr-multilingual-train-hn-mine", "es"),
    ]
    ds_tot = {}
    
    for (path, subset) in ds_paths:
        if subset is None:
            ds = cast(DatasetDict, load_dataset(path, num_proc=4))
            ds_tot = {**ds_tot, **ds}
        else:
            ds = cast(Dataset, load_dataset(path, subset, split="train", num_proc=4))
            ds_tot[f'{path.split("/")[1]}_{subset}'] = ds
            
    dataset = cast(DatasetDict, DatasetDict(ds_tot))

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    # dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    # print("Dataset size after filtering:", len(dataset))

    # keep only top 20 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:20]})

    # dataset_eval = dataset.select(range(500))
    # dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": None})

    corpus_paths = [
        ("manu/colpali-corpus", None),
        ("nomic-ai/vdr-multilingual-train-corpus", "it"),
        ("nomic-ai/vdr-multilingual-train-corpus", "en"),
        ("nomic-ai/vdr-multilingual-train-corpus", "fr"),
        ("nomic-ai/vdr-multilingual-train-corpus", "de"),
        ("nomic-ai/vdr-multilingual-train-corpus", "es"),
    ]
    anchor_ds_list = {}
    for (path, subset) in corpus_paths:
        if subset is None:
            ds = cast(Dataset, load_dataset(path, split="train", num_proc=4))
            anchor_ds_list[path.split("/")[1]] = ds
        else:
            ds = cast(Dataset, load_dataset(path, subset, split="train", num_proc=4))
            anchor_ds_list[subset] = ds

    anchor_ds = DatasetDict(anchor_ds_list)

    return ds_dict, anchor_ds, "vidore"

def load_train_set_vdr(lang: str = None) -> DatasetDict:
    if lang is None:
        ds_paths = [
            ("llamaindex/vdr-multilingual-train", "it"),
            ("llamaindex/vdr-multilingual-train", "en"),
            ("llamaindex/vdr-multilingual-train", "fr"),
            ("llamaindex/vdr-multilingual-train", "de"),
            ("llamaindex/vdr-multilingual-train", "es"),
        ]
    else:
        if lang not in ["fr", "es"]:
            ds_paths = [("llamaindex/vdr-multilingual-train", lang)]
        else:
            ds_paths = [("nomic-ai/vdr-multilingual-train", lang)]
    ds_tot = {}
    
    for (path, split) in ds_paths:
        if split is None:
            ds = cast(DatasetDict, load_dataset(path, num_proc=4))
            ds_tot = {**ds_tot, **ds}
        else:
            ds = cast(Dataset, load_dataset(path, split, split="train", num_proc=4))
            ds_tot[f'{path.split("/")[1]}_{split}'] = ds
            
    dataset = cast(DatasetDict, DatasetDict(ds_tot))

    dataset = dataset.rename_columns({"id": "image_filename"})
    dataset = dataset.remove_columns(["negatives"])
    
    ds_dict = DatasetDict({"train": dataset, "test": None})
    return ds_dict

def load_train_set_split_by_source() -> DatasetDict:
    ds_dict = cast(DatasetDict, load_dataset("nomic-ai/colpali_train_set_split_by_source"))
    return {"train": ds_dict, "test": None}


def load_train_set_detailed() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_train_set_with_tabfquad() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docmatix_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "docmatix-ir", split="train"))
    # dataset = dataset.select(range(100500))

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "Docmatix", "images", split="train"))

    return ds_dict, anchor_ds, "docmatix"

def load_wikiss() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "wiki-ss-nq", data_files="train.jsonl", split="train"))
    # dataset = dataset.select(range(400500))
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "wiki-ss-corpus", split="train"))

    return ds_dict, anchor_ds, "wikiss"


def load_train_set_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    dataset = cast(Dataset, load_dataset("nomic-ai/colpali-queries-mined-20250321-by-source"))

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    # dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    # print("Dataset size after filtering:", len(dataset))

    # keep only top 20 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:20]})

    # dataset_eval = dataset.select(range(500))
    # dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": None})

    anchor_ds = cast(Dataset, load_dataset(base_path + "colpali-corpus", split="train"))
    return ds_dict, anchor_ds, "vidore"


def load_train_set_with_docmatix() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
        "Docmatix_filtered_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot: List[Dataset] = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = concatenate_datasets(ds_tot)
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docvqa_dataset() -> DatasetDict:
    if USE_LOCAL_DATASET:
        dataset_doc = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="test"))
    else:
        dataset_doc = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test"))

    # concatenate the two datasets
    dataset = concatenate_datasets([dataset_doc, dataset_info])
    dataset_eval = concatenate_datasets([dataset_doc_eval, dataset_info_eval])
    # sample 100 from eval dataset
    dataset_eval = dataset_eval.shuffle(seed=42).select(range(200))

    # rename question as query
    dataset = dataset.rename_column("question", "query")
    dataset_eval = dataset_eval.rename_column("question", "query")

    # create new column image_filename that corresponds to ucsf_document_id if not None, else image_url
    dataset = dataset.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )
    dataset_eval = dataset_eval.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )

    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    return ds_dict


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test")
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
