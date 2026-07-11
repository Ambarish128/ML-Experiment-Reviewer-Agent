from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Dict

#Splitter - shared , used for text heavy documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def load_experiment_config(config_path:str)->str:
    """
    Loads a plain text experiment config file.
    Splits if necessary, then joins all chunks into one clean string
    for injection into a prompt.
    """
    loader=TextLoader(config_path)
    documents=loader.load()
    chunks = splitter.split_documents(documents)

    #Join all chunks - config needs to be read as a whole by the LLM
    full_text = "\n".join([chunk.page_content for chunk in chunks])
    return full_text

def load_experiment_metrics(metrics_path:str)->list[dict]:
    """
    Loads a CSV metrics file.
    Returns a list of dicts, one per epoch, for clean injection into prompts.
    """
    loader = CSVLoader(metrics_path)
    documents = loader.load()

    metrics=[]
    for doc in documents:
        # Each doc.page_content is a string like "epoch: 1\ntrain_loss: 0.85\n..."
        # Parse it back into a dict
        row={}
        for line in doc.page_content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":",1)
                row[key.strip()]=value.strip()
            
        metrics.append(row)

    return metrics

def load_experiment_artefacts(experiment_dir: str)->dict:
    """
    Master loader — given a directory, loads all artefacts for one experiment.
    Returns a single dict with everything chains need.
    """

    base = Path(experiment_dir)

    config_path=base/"exp_001_config.txt"
    metrics_path=base/"exp_001_metrics.csv"

    config_text = load_experiment_config(str(config_path))
    metrics = load_experiment_metrics(str(metrics_path))

    #Extract parallel lists for prompt injection
    train_losses =[float(row["train_loss"]) for row in metrics]
    val_losses = [float(row["val_loss"]) for row in metrics]
    eval_scores = [float(row["eval_score"]) for row in metrics]
    num_epochs = len(metrics)

    return {
        "config_text": config_text,
        "metrics": metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "eval_scores":eval_scores,
        "num_epochs": num_epochs,
        "experiment_id": "EXP_001"
    } 


# --------------------------------------------------
# Sanity check
# --------------------------------------------------
if __name__ == "__main__":
    artefacts = load_experiment_artefacts(".")

    print(f"Experiment ID   : {artefacts['experiment_id']}")
    print(f"Epochs loaded   : {artefacts['num_epochs']}")
    print(f"Train losses    : {artefacts['train_losses']}")
    print(f"Val losses      : {artefacts['val_losses']}")
    print(f"Eval scores     : {artefacts['eval_scores']}")
    print(f"\nConfig preview  :\n{artefacts['config_text'][:300]}")




