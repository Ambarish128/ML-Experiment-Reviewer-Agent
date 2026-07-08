from pydantic import BaseModel, Field
from typing import Literal, List

class ExperimentReview(BaseModel):
    experiment_id: str = Field(description="ID of the experiment under review")
    observation: str = Field(description="Objective summary of what the metrics show")
    concern_level: Literal["None", "Low", "Medium", "High"] = Field(
        description="Overall concern level based on the metrics"
    )
    recommendation: str = Field(description="One concrete, actionable next step")
    flags: List[str] = Field(
        default_factory=list,
        description="Specific anomalies spotted, empty list if none"
    )

class OverfittingReport(BaseModel):
    experiment_id: str = Field(description="ID of the experiment under review")
    overfitting_detected: bool = Field(description="Whether overfitting is present")
    onset_epoch: int | None = Field(
        default=None,
        description="Epoch where overfitting likely began, null if not detected"
    )
    severity: Literal["None", "Mild", "Moderate", "Severe"] = Field(
        description="Severity of overfitting"
    )
    reasoning: str = Field(description="Brief reasoning grounded in the metric values")

class HyperparameterReport(BaseModel):
    experiment_id: str = Field(description="ID of the experiment under review")
    issues_found: bool = Field(description="Whether any hyperparameter issues were found")
    flagged_params: List[str] = Field(
        default_factory=list,
        description="List of parameter names that look misconfigured"
    )
    explanation: str = Field(description="Explanation of what is wrong and why")
    suggested_values: str = Field(
        description="Concrete suggested corrections, or 'No changes needed'"
    )

    