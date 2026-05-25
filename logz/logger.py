import time

import wandb

log_times = []


def log(info, update_step, config):
    update_step = int(update_step)

    log_times.append(time.time())

    if len(log_times) == 1:
        print("Started logging")
    elif len(log_times) > 1:
        dt = log_times[-1] - log_times[-2]
        steps_between_updates = (
            config["NUM_STEPS"]
            * config["NUM_ENVS"]
            * config.get("WANDB_LOG_INTERVAL", 1)
        )
        sps = steps_between_updates / dt
        info["sps"] = sps

    # Use Python int instead of jnp.int32 so we don't overflow
    info["env_step"] = config["NUM_STEPS"] * config["NUM_ENVS"] * int(update_step)

    wandb.log(info)
