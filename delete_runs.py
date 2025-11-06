import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Optional


def delete_all_runs_in_experiment(experiment_name):
    """
    Delete all runs in an MLflow experiment given the experiment name.

    Args:
        experiment_name (str): Name of the experiment to delete runs from
    """
    client = MlflowClient()

    try:
        # Get the experiment by name
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return

        experiment_id = experiment.experiment_id
        print(f"Found experiment '{experiment_name}' with ID: {experiment_id}")

        # Get all runs in the experiment (including deleted ones)
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=10000,
        )

        if not runs:
            print(f"No runs found in experiment '{experiment_name}'.")
            return

        print(f"Found {len(runs)} runs in experiment '{experiment_name}'.")

        # Delete each run
        deleted_count = 0
        for run in runs:
            try:
                client.delete_run(run.info.run_id)
                deleted_count += 1
                print(f"Deleted run: {run.info.run_id}")
            except Exception as e:
                print(f"Failed to delete run {run.info.run_id}: {e}")

        print(f"Successfully deleted {deleted_count} out of {len(runs)} runs.")

    except Exception as e:
        print(f"Error: {e}")


def get_child_runs(parent_run_name: str, experiment_name: Optional[str] = None) -> List[mlflow.entities.Run]:
    """
    Get all child runs associated with a parent run given the parent run name.

    Args:
        parent_run_name: Name of the parent run
        experiment_name: Optional experiment name to search within. If None, searches all experiments.

    Returns:
        List of child runs
    """
    client = mlflow.tracking.MlflowClient()

    # Find the parent run by name
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        experiment_ids = [experiment.experiment_id]
    else:
        # Search across all experiments
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]

    # Search for the parent run by name
    parent_run = None
    for exp_id in experiment_ids:
        runs = client.search_runs(experiment_ids=[exp_id], filter_string=f"tags.mlflow.runName = '{parent_run_name}'")
        if runs:
            parent_run = runs[0]
            break

    if not parent_run:
        raise ValueError(f"Parent run with name '{parent_run_name}' not found")

    # Get all child runs
    child_runs = client.search_runs(
        experiment_ids=experiment_ids, filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
    )

    return [child_run.info.run_id for child_run in child_runs]


def delete_failed_temperature_runs(experiment_name: Optional[str] = None) -> None:
    """
    Delete parent runs whose names start with "temperature-" and have failed status,
    along with all their child runs.

    Args:
        experiment_name: Optional experiment name to search within. If None, searches all experiments.
    """
    client = MlflowClient()

    try:
        # Get experiment IDs to search
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            experiment_ids = [experiment.experiment_id]
            print(f"Searching in experiment '{experiment_name}'")
        else:
            # Search across all experiments
            experiments = client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]
            print("Searching across all experiments")

        failed_temperature_runs = []

        for status in ["FAILED", "RUNNING"]:
            for exp_id in experiment_ids:
                failed_temperature_runs = client.search_runs(
                    experiment_ids=[exp_id],
                    # Filter runs that have failed status and start with "temperature="
                    filter_string=f"attribute.status = '{status}' AND tags.mlflow.runName LIKE 'temperature=%'",
                    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                    max_results=10000,
                )

            if not failed_temperature_runs:
                print(f"No {status} parent runs found.")

            else:
                print(f"Found {len(failed_temperature_runs)} {status} parent runs.")

                total_deleted = 0
                # For each failed temperature parent run, delete it and its children
                for parent_run in failed_temperature_runs:
                    parent_run_id = parent_run.info.run_id
                    parent_run_name = parent_run.data.tags.get("mlflow.runName", "")

                    print(f"Processing parent run: {parent_run_name} (ID: {parent_run_id})")

                    # Find all child runs
                    child_runs = client.search_runs(
                        experiment_ids=experiment_ids,
                        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                        max_results=10000,
                    )

                    # Delete child runs first
                    child_deleted_count = 0
                    for child_run in child_runs:
                        try:
                            client.delete_run(child_run.info.run_id)
                            child_deleted_count += 1
                            print(f"  Deleted child run: {child_run.info.run_id}")
                        except Exception as e:
                            print(f"  Failed to delete child run {child_run.info.run_id}: {e}")

                    # Delete the parent run
                    try:
                        client.delete_run(parent_run_id)
                        print(f"  Deleted parent run: {parent_run_name} (ID: {parent_run_id})")
                        total_deleted += 1 + child_deleted_count
                    except Exception as e:
                        print(f"  Failed to delete parent run {parent_run_id}: {e}")

                print(f"Successfully deleted {total_deleted} runs total (parents + children).")

    except Exception as e:
        print(f"Error: {e}")


def log_best_loss(experiment_name: str) -> None:
    """
    Loop over each parent run, check the loss history, and log the best value as a new metric.
    """
    client = MlflowClient()

    try:
        # Get the experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search for all parent runs i.e. runs starting with "temperature-"
        parent_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName LIKE 'temperature=%'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=10000,
        )

        for parent_run in parent_runs:
            # Get the loss history for the parent run
            loss_history = client.get_metric_history(parent_run.info.run_id, "loss")

            if loss_history:
                # Find the best loss value
                best_loss = min(loss_history, key=lambda x: x.value)
                print(f"Best loss for parent run {parent_run.info.run_id}: {best_loss.value}")

                # Log the best loss as a new metric
                client.log_metric(parent_run.info.run_id, "best_loss", best_loss.value)

    except Exception as e:
        print(f"Error: {e}")


# Usage example based on your config
if __name__ == "__main__":
    # Delete all runs in the experiment
    experiment_name = "arbitrary-64lines-more"  # From your config
    # delete_all_runs_in_experiment(experiment_name)
    delete_failed_temperature_runs(experiment_name)
