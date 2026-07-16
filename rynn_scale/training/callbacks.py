import os

from transformers import TrainerCallback
from transformers.integrations.integration_utils import INTEGRATION_TO_CALLBACK


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class MLTrackerTrainerCallback(TrainerCallback):
    def __init__(self):
        import ml_tracker

        self._ml_tracker = ml_tracker
        self._initialized = False

    def setup(self, args, state, model, **kwargs):
        if self._ml_tracker is None:
            return
        self._initialized = True

        if state.is_world_process_zero:
            combined_dict = {**args.to_dict()}
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            if hasattr(model, "peft_config") and model.peft_config is not None:
                peft_config = model.peft_config
                combined_dict = {**{"peft_config": peft_config}, **combined_dict}

            if self._ml_tracker.run is None:
                self._ml_tracker.init(project=os.getenv("ML_TRACKER_PROJECT", None))
            # add config parameters (run may have been created manually)
            self._ml_tracker.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest ml_tracker versions)
            if getattr(self._ml_tracker, "define_metric", None):
                self._ml_tracker.define_metric("train/global_step")
                self._ml_tracker.define_metric("*", step_metric="train/global_step", step_sync=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._ml_tracker is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._ml_tracker.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._ml_tracker is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._ml_tracker.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._ml_tracker.log({**non_scalar_logs, "train/global_step": state.global_step}, step=state.global_step)

    def on_predict(self, args, state, control, metrics, **kwargs):
        if self._ml_tracker is None:
            return
        if not self._initialized:
            self.setup(args, state, **kwargs)
        if state.is_world_process_zero:
            metrics = rewrite_logs(metrics)
            self._ml_tracker.log(metrics)


INTEGRATION_TO_CALLBACK["ml_tracker"] = MLTrackerTrainerCallback
