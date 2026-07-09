from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from prompts import experiment_review_prompt, overfitting_detection_prompt, hyperparameter_prompt
from schemas import ExperimentReview, OverfittingReport, HyperparameterReport

load_dotenv()

## LLM - single shared instance across all chains

llm = ChatGroq(model = "openai/gpt-oss-120b", temperature = 0)

#Parsers

expriment_review_parser = PydanticOutputParser(pydantic_object=ExperimentReview)
overfitting_parser = PydanticOutputParser(pydantic_object=OverfittingReport)
hyperparameter_parser = PydanticOutputParser(pydantic_object=HyperparameterReport)

# Chains

experiment_review_chain = experiment_review_prompt | llm | expriment_review_parser
overfitting_chain = overfitting_detection_prompt | llm | overfitting_parser
hyperparameter_chain = hyperparameter_prompt | llm | hyperparameter_parser

#Sanity Check

if __name__ == "__main__":
    
    print("\n--- Overfitting Chain ---")
    result = overfitting_chain.invoke({
        "experiment_id": "exp_001",
        "train_losses": [0.85, 0.60, 0.42, 0.31, 0.24],
        "val_losses":   [0.88, 0.65, 0.61, 0.74, 0.91],
        "num_epochs": 5
    })
    print(result.model_dump())

    print("\n--- Hyperparameter Chain ---")
    result = hyperparameter_chain.invoke({
        "experiment_id": "exp_001",
        "architecture": "3-layer MLP",
        "hyperparameters": {"lr": 10.0, "dropout": 0.0},
        "dataset_size": 500,
        "task_type": "binary classification"
    })
    print(result.model_dump())

    print("\n--- Experiment Review Chain ---")
    result = experiment_review_chain.invoke({
        "experiment_id": "exp_001",
        "architecture": "3-layer MLP",
        "train_losses": [0.85, 0.60, 0.42, 0.31, 0.24],
        "val_losses":   [0.88, 0.65, 0.61, 0.74, 0.91],
        "eval_metric": "F1",
        "eval_score": 0.61,
        "num_epochs": 5,
        "hyperparameters": {"lr": 0.01, "batch_size": 32, "dropout": 0.0}
    })
    print(result.model_dump())
