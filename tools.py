from langchain_core.tools import tool

@tool
def summarise_experiment(experiment_id: str, train_losses: str, val_losses: str, num_epochs: str) -> str:
    """
    Summarise training and validation loss history for an ML experiment.
    Use this when you need a human-readable overview of a training run —
    best/worst validation loss, which epoch each occurred, and overall trend.
    Input losses as comma-separated strings e.g. '0.85,0.60,0.42'.
    """
    train = [float(x.strip()) for x in train_losses.split(",") if x.strip()]
    val   = [float(x.strip()) for x in val_losses.split(",") if x.strip()]
    epochs = int(num_epochs)

    best_loss   = min(val)
    best_epoch  = val.index(best_loss) + 1
    worst_loss  = max(val)
    worst_epoch = val.index(worst_loss) + 1

    improving = all(val[i] <= val[i - 1] for i in range(1, len(val)))
    degrading = all(val[i] >= val[i - 1] for i in range(1, len(val)))
    trend = "Improving" if improving else "Degrading" if degrading else "Mixed"

    return (
        f"Experiment Summary\n"
        f"------------------\n"
        f"Experiment ID : {experiment_id}\n"
        f"Total Epochs  : {epochs}\n"
        f"Best Val Loss : {best_loss:.4f} (Epoch {best_epoch})\n"
        f"Worst Val Loss: {worst_loss:.4f} (Epoch {worst_epoch})\n"
        f"Overall Trend : {trend}"
    )

@tool
def detect_overfitting_signal(train_losses: str, val_losses: str) -> str:
    """
    Detect overfitting by comparing training and validation loss at each epoch.
    Flags any epoch where validation loss exceeds training loss by more than 15%.
    Use this when you have per-epoch loss values and need to determine if and
    when overfitting began. Input losses as comma-separated strings.
    Returns flagged epochs and a final verdict.
    """
    train = [float(x.strip()) for x in train_losses.split(",") if x.strip()]
    val   = [float(x.strip()) for x in val_losses.split(",") if x.strip()]

    if len(train) != len(val):
        raise ValueError("Training and validation loss lists must be the same length.")

    flagged_epochs = [
        i + 1 for i in range(len(val))
        if (val[i] - train[i]) / train[i] > 0.15
    ]

    verdict = "Overfitting signal detected" if flagged_epochs else "No overfitting signal detected"
    return f"Flagged epochs: {flagged_epochs}\nVerdict: {verdict}"

@tool
def check_hyperparameter(param_name: str, value: str) -> str:
    """
    Check whether a single hyperparameter value falls within a known-good range.
    Use this when you want to verify if a specific hyperparameter looks misconfigured.
    Input param_name as a string e.g. 'learning_rate' and value as a numeric string.
    Supported params: learning_rate, dropout, batch_size, weight_decay.
    Returns whether the value is safe or suspicious and why.
    """
    safe_ranges = {
        "learning_rate": (1e-5, 1e-1),
        "dropout":       (0.0, 0.7),
        "batch_size":    (8, 512),
        "weight_decay":  (0.0, 0.1)
    }

    try:
        val = float(value.strip())
    except ValueError:
        return f"Could not parse value '{value}' as a number."

    param = param_name.strip().lower()
    if param not in safe_ranges:
        return f"No known range for '{param}'. Cannot assess."

    low, high = safe_ranges[param]
    if low <= val <= high:
        return f"{param}={val} is within the normal range [{low}, {high}]. No issue."
    return f"WARNING: {param}={val} is outside the normal range [{low}, {high}]. Likely misconfigured."

# All tools in one list — imported by the agent whenever required
all_tools=[summarise_experiment,detect_overfitting_signal,check_hyperparameter]

# --------------------------------------------------
# Sanity check
# --------------------------------------------------
if __name__ == "__main__":
    train = "1.842,1.286,0.934,0.712,0.548,0.421,0.316,0.241,0.188,0.142"
    val   = "1.913,1.331,0.972,0.756,0.792,0.948,1.183,1.472,1.826,2.214"

    print(summarise_experiment.invoke({
        "experiment_id": "EXP_001",
        "train_losses": train,
        "val_losses": val,
        "num_epochs": "10"
    }))

    print("\n" + "="*50 + "\n")

    print(detect_overfitting_signal.invoke({
        "train_losses": train,
        "val_losses": val
    }))

    print("\n" + "="*50 + "\n")

    print(check_hyperparameter.invoke({"param_name": "learning_rate", "value": "10.0"}))
    print(check_hyperparameter.invoke({"param_name": "dropout", "value": "0.5"}))