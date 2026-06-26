import subprocess
import argparse
import warnings
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
from src.utils.torch_utils import get_device, get_net_trainable_params
from src.data.builder import build_dataset, build_dataloaders
from src.models.builder import build_model
from src.utils import set_random_seed
from src.utils.config_files_utils import override_config, read_yaml

# --------------------------------------------------------------------------------------
#                   Don't use this file, it has to be reviewed
# --------------------------------------------------------------------------------------

# To monitor training/inference time
# https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch


def set_clock_speed():
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling,
    but can help reduce variance.
    """
    process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    process = subprocess.run(
        f"sudo nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}", shell=True
    )


def reset_clock_speed():
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {DEVICE}", shell=True)


# TODO: was quite modified, check eveything works properly
def benchmark(net, optimizer, dataloader, warmup_steps, kernel, device):
    """Infos:
    Dataloader pas de type UDA
    """
    # L2 cache size on A100
    x = torch.empty(int(40 * (1024**2)), dtype=torch.int8, device="cuda")

    def flush_cache():
        x.zero_()

    set_clock_speed()

    iterator = iter(dataloader)
    dataloader_length = dataloader.__len__()
    assert warmup_steps < dataloader_length, "incorrect number of warmup steps"
    pbar = tqdm(
        total=dataloader_length + warmup_steps,
        desc="Benchmarking",
        position=0,
        leave=True,
    )

    for _ in range(warmup_steps):
        sample = next(iterator)
        kernel(net, sample, device)
        pbar.update(1)

    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(dataloader_length)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(dataloader_length)
    ]

    iterator = iter(dataloader)

    for i in range(dataloader_length):
        sample = next(iterator)
        flush_cache()
        torch.cuda._sleep(1_000_000)

        start_events[i].record()
        kernel(net, sample, device)
        end_events[i].record()
        pbar.update(1)

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    reset_clock_speed()
    return times


if __name__ == "__main__":
    # vérifier dimension squeeze labels
    parser = argparse.ArgumentParser(description="PyTorch self-training")
    parser.add_argument(
        "-c", "--config", help="configuration (.yaml) file to use", metavar="FILE"
    )
    parser.add_argument("--device", default="0,1", type=str, help="gpu ids to use")
    parser.add_argument("--warmup", default=10, type=int, help="number of warup steps")
    parser.add_argument("--wandb", help="send results to wandb", action="store_true")
    # parser.add_argument(
    #     "--lin", action="store_true", help="train linear classifier only"
    # )
    # lin_cls = args.lin

    args, config = override_config(parser)

    print(args.device)
    device_ids = [int(d) for d in args.device.split(",")]
    device = get_device(device_ids, allow_cpu=False)
    config["local_device_ids"] = device_ids

    # Create work dir
    workdirp = Path(config["CHECKPOINT"]["save_path"])
    if workdirp.exists():
        warnings.warn(
            "directory %s already exists, original content will be override" % workdirp
        )
    workdirp.mkdir(parents=True, exist_ok=True)

    # Logger
    # logger_config = read_yaml("./logging.yaml")
    # logger_config["handlers"]["file"]["filename"] = Path(
    #     config["CHECKPOINT"]["save_path"], "log.txt"
    # )
    # logging.config.dictConfig(logger_config)
    # logger = logging.getLogger("uda_train")

    set_random_seed(config["RUNTIME"]["seed"], config["RUNTIME"]["deterministic"])

    DATASET_INFO = read_yaml(config["DATA"]["DATASET_INFO"])
    print(DATASET_INFO)
    model_config = config["MODEL"]
    cfg = config["DATA"].get("benchmark")
    print(cfg)
    dataset = {"benchmark": build_dataset(cfg, "benchmark", model_config, DATASET_INFO)}
    dataloader = build_dataloaders(dataset, config)
    net = build_model(model_config)
    trainable_params = get_net_trainable_params(net)
    # Normally the following  hardcoded values do not have any kind of influence
    optimizer = optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.0)
    if args.wandb:
        import wandb

        root_logdir = config["CHECKPOINT"]["save_path"]
        wandb.login()
        run = wandb.init(
            project=config["WANDB"]["project"],
            name=config["WANDB"]["name"],
            config=config,
            dir=workdirp,
        )
    # train_step_args = {"net":net, "teacher":teacher, "optimizer":optimizer, "loss_input_fn":loss_input_fn, "alpha":alpha, "device":device}
    # train_step = build_train_step(config,**train_step_args)

    times = benchmark(net, optimizer, dataloader, args.warmup, kernel, device)

    if args.wandb:
        run.finish()
