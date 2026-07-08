from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import ExperimentReview, OverfittingReport, HyperparameterReport

#Creating parsers for every use-case

experiment_review_parser = PydanticOutputParser(pydantic_object=ExperimentReview)
overfitting_parser = PydanticOutputParser(pydantic_object=OverfittingReport)
hyperparameter_parser = PydanticOutputParser(pydantic_object=HyperparameterReport)



# General Experiment Review Prompt

experiment_review_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a senior ML engineer conducting structured peer reviews of ML experiments.
You are precise, technical, and evidence-driven. You do not speculate beyond what the metrics show.
Always structure your response in exactly three sections:
- Observation: what the metrics objectively show
- Concern Level: None / Low / Medium / High
- Recommendation: one concrete next step
     
     {format_instructions}"""),
    ("human","""Review the following experiment:

Experiment ID: {experiment_id}
Model Architecture: {architecture}
Training Loss (per epoch): {train_losses}
Validation Loss (per epoch): {val_losses}
Evaluation Metric: {eval_metric}
Eval Score: {eval_score}
Number of Epochs: {num_epochs}
Hyperparameters: {hyperparameters}

Provide your structured review."""),
]).partial(format_instructions= experiment_review_parser.get_format_instructions())

# Prompt for Overfitting Detection

overfitting_detection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an ML specialist focused exclusively on detecting overfitting in training runs.
You do not provide general experiment feedback.
Assess only whether overfitting is present, when it began, and how severe it is.
Severity scale: None / Mild / Moderate / Severe.
     
     {format_instructions}"""),
    ("human", """Analyse the following training run for overfitting:

Experiment ID: {experiment_id}
Training Losses (per epoch): {train_losses}
Validation Losses (per epoch): {val_losses}
Number of Epochs: {num_epochs}

Determine:
1. Is overfitting present?
2. If yes, at which epoch did it likely begin?
3. Severity: None / Mild / Moderate / Severe
4. Brief reasoning grounded in the numbers.""")
]).partial(format_instructions=overfitting_parser.get_format_instructions())

#Prompt for Hyperparameter Sanity Check

hyperparameter_prompt = ChatPromptTemplate.from_messages([
    ("system","""You are an ML engineer who specialises in hyperparameter analysis.
Given a model architecture and its hyperparameters, identify anything that looks misconfigured, 
unusual, or likely to cause training instability. Be specific — cite the actual values.
     
     {format_instructions}"""),
    ("human","""Check the following hyperparameter configuration:

Experiment ID: {experiment_id}
Model Architecture: {architecture}
Hyperparameters: {hyperparameters}
Dataset Size: {dataset_size}
Task Type: {task_type}

Flag any values that look incorrect or suspicious, and explain why.""")
]).partial(format_instructions = hyperparameter_parser.get_format_instructions())

# Sanity Check to see if prompts work as expected,(run this file to verify)

if __name__ == "__main__":
    test_cases = [
        (experiment_review_prompt, {
            "experiment_id": "exp_001",
            "architecture": "3-layer MLP",
            "train_losses": [0.85, 0.60, 0.42, 0.31, 0.24],
            "val_losses":   [0.88, 0.65, 0.61, 0.74, 0.91],
            "eval_metric": "F1",
            "eval_score": 0.61,
            "num_epochs": 5,
            "hyperparameters": {"lr": 0.01, "batch_size": 32, "dropout": 0.0}
        }),
        (overfitting_detection_prompt, {
            "experiment_id": "exp_001",
            "train_losses": [0.85, 0.60, 0.42, 0.31, 0.24],
            "val_losses":   [0.88, 0.65, 0.61, 0.74, 0.91],
            "num_epochs": 5
        }),
        (hyperparameter_prompt, {
            "experiment_id": "exp_001",
            "architecture": "3-layer MLP",
            "hyperparameters": {"lr": 0.01, "batch_size": 32, "dropout": 0.0},
            "dataset_size": 500,
            "task_type": "binary classification"
        }),
    ]

    for i, (prompt, values) in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"PROMPT {i}")
        print('='*50)
        messages = prompt.format_messages(**values)
        for message in messages:            
            print(f"[{message.type.upper()}]:\n{message.content}\n")
            break

