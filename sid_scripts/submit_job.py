# launch_with_nemorun.py
import os
from os.path import expanduser
import nemo_run as run

# Host <-> container mounts
mounts = [
    "/lustre/fsw/portfolios/llmservice/users/igitman/hf_models:/hf_models",
    "/lustre/fsw/portfolios/llmservice/users/igitman/nemo_models:/nemo_models",
    "/lustre/fsw/portfolios/llmservice/users/siddjain/workspace:/workspace",
    "/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output:/output",
    "/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/archive:/archive",
    "/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data:/data",
    "/lustre/fsw/portfolios/llmservice/users/siddjain/my_models:/my_models",
    "/lustre:/lustre",
    "/home/siddjain/.netrc:/root/.netrc",
]

@run.cli.entrypoint
def main(
    nodes: int = 1,
    gpus_per_node: int = 8,
    time: str = "04:00:00",
    partition: str = "interactive",
    account: str = "llmservice_nemo_reasoning",
    job_name_prefix: str = "llmservice_nemo_reasoning-math-retrieval",
    container_image: str = "/lustre/fsw/portfolios/llmservice/users/siddjain/containers/citer_v1.sqsh",
    workdir_subpath: str = "",  # optional: subdir of your git repo to run from
    extra_env: dict | None = None,
    # SSH tunnel params
    ssh_host: str = "cw-dfw-cs-001-login-02.cw-dfw-cs-001.hpc.nvidia.com",
    ssh_user: str = "siddjain",
    ssh_key_path: str = "/Users/siddjain/.ssh/id_rsa",
):
    """
    Submit your training via NeMo-Run + Slurm + torchrun, tunneling over SSH.
    """

    # Package the current git repo (optionally a subpath) so remote workers get the same code snapshot.
    packager = run.GitArchivePackager(subpath=workdir_subpath or "")

    # SSH tunnel (NeMo-Run expects 'identity' and a remote job_dir; 'port' is not a kw)
    tunnel = run.SSHTunnel(
        host=ssh_host,
        user=ssh_user,
        identity=expanduser(ssh_key_path),
        job_dir="/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run",
    )

    # Slurm executor; torchrun world-size is inferred from nodes/ntasks_per_node
    executor = run.SlurmExecutor(
        account=account or None,
        partition=partition,
        job_name_prefix=f"{job_name_prefix}:",
        container_image=container_image,
        container_mounts=mounts,
        time=time,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        ntasks_per_node=gpus_per_node,  # 1 task per GPU
        tunnel=tunnel,
        env_vars={
            "PYTHONUNBUFFERED": "1",
            "NCCL_DEBUG": "WARN",
            # Add cluster-specific NCCL envs here if needed, e.g.:
            # "NCCL_IB_HCA": "mlx5",
            # "NCCL_SOCKET_IFNAME": "eth0",
            **(extra_env or {}),
        },
        packager=packager,
    )

    # Use Torchrun launcher; rendezvous is derived under Slurm
    executor.launcher = run.Torchrun()

    # Define the task to run inside the container.
    # Script(config) uses 'inline' or 'path' (no 'cmd'); keep it a single-line shell command.
    task = run.Script(
        inline="cd /workspace/CITER && python train_cluster.py"
    )

    run.run(task, executor)

if __name__ == "__main__":
    run.cli.main(main)
