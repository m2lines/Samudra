import argparse
import logging
import os
import shlex
import shutil
import sys
import traceback
from pathlib import Path
from re import S
from typing import Any, Dict, List, Sequence

import hydra
import numpy as np
import submitit
import submitit.slurm.slurm as submitit_slurm
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path
from hydra.core.plugins import Plugins
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, filter_overrides
from hydra.experimental.callback import Callback
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import \
    BaseQueueConf
from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger(__name__)
CONFIG_PATH = "configs"
NAME_MAX = 255

@hydra.main(version_base=None, config_name="conf", config_path=CONFIG_PATH)
def my_app(cfg: DictConfig) -> None:
    try:
        env = submitit.JobEnvironment()
        log.info(env.__repr__())
    except:
        log.info("Running locally.")
    
    # OmegaConf.set_struct(cfg, False)
    sys.path.append("src/")
    main_fn = __import__("src." + cfg.experiment, fromlist=[None]).main
    log.info(f"Beginning experiment [{cfg.experiment}].")
    main_fn(cfg)
    log.info(f"Completed experiment.")

class LogJobReturnCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            e_str = "".join(traceback.format_exception(None, # <- type(e) by docs, but ignored 
                                                                job_return._return_value,
                                                                job_return._return_value.__traceback__))
            # We use sys.stderr to ensure that it always prints as some code overwrites print.
            sys.stderr.write(e_str) 
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")

if __name__ == "__main__":
    # Omegaconf resolvers for managing configs.
    OmegaConf.register_new_resolver("prod", lambda *args: np.prod(args).item()) # product arguments
    OmegaConf.register_new_resolver("replace_slash", lambda s: s.replace("/", ".")) # replaces slashes in str
    OmegaConf.register_new_resolver("join_path", lambda *args: os.path.join(*args)) # joins paths
    OmegaConf.register_new_resolver("join_overlays", lambda *args: ','.join(filter(None, args))) # joins overlays")
    OmegaConf.register_new_resolver("limit_path_length", lambda s: s[:NAME_MAX]) # limits path length
    
    # [HACK] 1 
    # We overwrite the _submitit_command_str function to use a alternative script that will:
    # (1) load infiniband configs and run the command inside a preconfigured singularity container
    # (2) redirect the initial __main__ function to a custom one that sets up code for automatic resubmission
    # By setting this, submitit will use this alternative to produce the SBATCH script.
    @property
    def _submitit_command_str(self) -> str:
        return " ".join(
            ["--cpu-bind=verbose", "\\\n",
             shlex.quote(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".resubmit.sh")), "\\\n",
             shlex.quote(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".python-greene")), "\\\n",
             "-u -m submitit.core._submit",
             shlex.quote(str(self.folder))]
        )
    submitit_slurm.SlurmExecutor._submitit_command_str = _submitit_command_str
    
    # [HACK] 2
    # We overwrite the submit command as well. We allow it to specify an array and cpus-per-task 
    # command line argument. Note, cpus-per-task and mem are not typically useful, however they're used
    # during resubmission.
    # This allows us to:
    # (1) selectively re-execute based on the array values
    # (2) easily re-queue jobs with this to continue previously existing jobs by using the exact same args.
    # Note that this MUST be done before the hydra.main decorated fn as this is used for SLURM and using the
    # submitit argument means no logic can be run between the end of this function and execution on a worker.
    # [HACK] 2b: We must also remove the --array flag from sys.argv else hydra argparse will error.
    # This is done via sys.argv = sys.argv[:1] + replace (replace is the unused flags from our parser)
    parser = argparse.ArgumentParser()
    parser.add_argument("--array", type=str, required=False)
    parser.add_argument("--cpus-per-task", type=str, required=False)
    parser.add_argument("--mem", type=str, required=False)
    args, replace = parser.parse_known_args()
    sys.argv = sys.argv[:1] + replace # we may further modify argv to prevent creating extra directories that aren't run
    def _make_submission_command(self, submission_file_path: Path) -> List[str]:
        command_list = ["sbatch", str(submission_file_path)]
        if args.mem is not None:
            command_list.insert(1, f"--mem={args.mem}")
        if args.cpus_per_task is not None:
            command_list.insert(1, f"--cpus-per-task={args.cpus_per_task}")
        if args.array is not None:
            command_list.insert(1, f"--array={args.array}")
        return command_list
    submitit_slurm.SlurmExecutor._make_submission_command = _make_submission_command
    
    # [HACK] 3
    # We need to modify the SlurmExecutor with a wrapper to copy code into the correct job directory once the job has been submitted.
    # This is so that our job is not impacted by the state of the code when it eventually gets allocated resources.
    # We expect that at the time of job submission $PWD is equal to where the experiment code is stored. (regardless of where this file is located)
    def _internal_process_submissions_wrapper(func):
        def fn(self, delayed_submissions):
            jobs = func(self, delayed_submissions)
            source_path = os.getcwd()
            copy_path = os.path.join(str(self.folder).rsplit("/", 2)[0], ".snapshot") # per-experiment batch in $output_dir/.src
            if not os.path.exists(copy_path): # do not recopy.
                os.makedirs(copy_path, exist_ok=False)
                shutil.copy(os.path.join(source_path, "submitit_hydra.py"), 
                            os.path.join(copy_path, "submitit_hydra.py"))
                shutil.copytree(os.path.join(source_path, "src"), 
                                os.path.join(copy_path, "src"))
                shutil.copytree(os.path.normpath(os.path.join(os.getcwd(), CONFIG_PATH)), 
                                os.path.join(copy_path, "configs"))
            return jobs
        return fn
    submitit_slurm.SlurmExecutor._internal_process_submissions = (
        _internal_process_submissions_wrapper(submitit_slurm.SlurmExecutor._internal_process_submissions)
    )
    
    # [HACK] 4
    # This one's an absolute nightmare of a hack to address an actual bug in submitit-hydra
    # Upon running on a SLURM worker node, the submitit launcher attempts to re-read the configs even if they have changed.
    # My solution to this is to snapshot my configs and redirect the re-load to this new, snapshotted location.
    # The actual fix would likely to be replace L62-L64 with something that does not re-load the entire config tree. LMFAO
    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}

        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        log.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : "
            f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )
        
        job_dir = str(OmegaConf.select(self.config, job_params[0][1]))
        self.config.hydra.runtime.config_sources[1].path = os.path.join(job_dir, ".snapshot", "configs")
        self.hydra_context.config_loader = ConfigLoaderImpl(create_config_search_path(os.path.join(job_dir, ".snapshot", "configs")))
        
        jobs = executor.map_array(self, *zip(*job_params))
        return [j.results()[0] for j in jobs]
    Plugins.instance().class_name_to_class['hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher'].launch = launch

    my_app()
