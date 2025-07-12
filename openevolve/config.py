"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(default_factory=lambda: [LLMModelConfig()])

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if (self.primary_model or self.primary_model_weight) and len(self.models) < 1:
            # Ensure we have a primary model
            self.models.append(LLMModelConfig())
        if self.primary_model:
            self.models[0].name = self.primary_model
        if self.primary_model_weight:
            self.models[0].weight = self.primary_model_weight

        if (self.secondary_model or self.secondary_model_weight) and len(
            self.models
        ) < 2:
            # Ensure we have a second model
            self.models.append(LLMModelConfig())
        if self.secondary_model:
            self.models[1].name = self.secondary_model
        if self.secondary_model_weight:
            self.models[1].weight = self.secondary_model_weight

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
        }
        self.update_model_params(shared_config)

    def update_model_params(
        self, args: Dict[str, Any], overwrite: bool = False
    ) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["score", "complexity"]
    )
    feature_bins: int = 10

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution"""

    # General sandbox settings
    enabled: bool = False
    timeout: int = 600  # 10 minutes per sandbox
    executor_idle_timeout: int = 3600  # 1 hour idle timeout for executor

    # Resource limits
    cpu_limit: float = 0.25  # CPU limit to prevent resource exhaustion
    memory_limit_mb: int = 512  # Memory limit in MiB

    # Security settings
    block_network: bool = True  # Block network access for security
    working_directory: str = "/workspace"  # Working directory inside sandbox

    # Volume configuration
    evaluation_volume_path: str = "/eval"  # Path to mount evaluation volume

    # Default dependencies for sandbox image
    default_dependencies: List[str] = field(default_factory=lambda: ["pytest", "numpy"])

    # Concurrency settings
    max_concurrent_evaluations: int = 64
    target_concurrent_evaluations: int = 32


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation (legacy - moved to SandboxConfig)
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 4
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1
    llm_evaluation_template: str = "evaluation"  # Template key for LLM evaluation

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program

    # Artifact keys configuration
    artifact_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "timeout": "timeout",
            "timeout_duration": "timeout_duration",
            "failure_stage": "failure_stage",
            "error_type": "error_type",
            "stderr": "stderr",
            "traceback": "traceback",
            "stage1_timeout": "stage1_timeout",
            "stage2_timeout": "stage2_timeout",
            "stage3_timeout": "stage3_timeout",
            "stage2_stderr": "stage2_stderr",
            "stage2_traceback": "stage2_traceback",
            "stage3_stderr": "stage3_stderr",
            "stage3_traceback": "stage3_traceback",
        }
    )

    # Sandboxed execution
    use_sandboxed_execution: bool = False  # Use Modal sandboxes for evaluation
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    # File handling
    temp_file_suffix: str = ".py"  # Suffix for temporary evaluation files

    # JSON parsing configuration
    json_extract_patterns: List[str] = field(
        default_factory=lambda: [
            r"```json\n(.*?)\n```",  # JSON code blocks
            r"\{.*\}",  # Direct JSON objects
        ]
    )


@dataclass
class ModalConfig:
    """Configuration for Modal distributed execution"""

    # General Modal settings
    enabled: bool = False
    app_name: str = "openevolve"

    # Controller Hub settings
    hub_timeout: int = 3600  # 1 hour idle timeout
    hub_max_containers: int = 1  # Single-writer semantics
    hub_max_concurrent_requests: int = 999

    # Evolution worker settings
    worker_min_containers: int = 0
    worker_max_containers: int = 400
    worker_timeout: int = 900  # 15 minutes
    worker_buffer_size: int = 100  # max active containers per generation

    # LLM generation settings
    llm_min_containers: int = 0
    llm_max_containers: int = 50  # CPU-only for HTTP calls
    llm_timeout: int = 600  # 10 minutes

    # Evaluation settings
    eval_min_containers: int = 0
    eval_max_containers: int = 256
    eval_timeout: int = 600  # 10 minutes
    eval_retries: int = 3
    eval_max_concurrent_inputs: int = 64
    eval_target_concurrent_inputs: int = 32

    # Volume paths
    database_volume_path: str = "/db"
    evaluation_volume_path: str = "/eval"

    # Checkpointing
    checkpoint_interval_generations: int = 10
    export_best_interval_generations: int = 5

    # Secret names
    inference_secret_name: str = "inference-secret"  # TODO: currently unused

    # Resume configuration
    resume_from_checkpoint: bool = True

    # Status polling
    status_poll_interval: float = 2.0  # seconds between status checks


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in [
                "llm",
                "prompt",
                "database",
                "evaluator",
                "modal",
            ] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            evaluator_dict = config_dict["evaluator"]
            # Handle nested SandboxConfig
            if "sandbox" in evaluator_dict:
                evaluator_dict["sandbox"] = SandboxConfig(**evaluator_dict["sandbox"])
            config.evaluator = EvaluatorConfig(**evaluator_dict)
        if "modal" in config_dict:
            config.modal = ModalConfig(**config_dict["modal"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                "include_artifacts": self.prompt.include_artifacts,
                "max_artifact_bytes": self.prompt.max_artifact_bytes,
                "artifact_security_filter": self.prompt.artifact_security_filter,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                "diversity_metric": self.database.diversity_metric,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
                "artifacts_base_path": self.database.artifacts_base_path,
                "artifact_size_threshold": self.database.artifact_size_threshold,
                "cleanup_old_artifacts": self.database.cleanup_old_artifacts,
                "artifact_retention_days": self.database.artifact_retention_days,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                "memory_limit_mb": self.evaluator.memory_limit_mb,
                "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
                "llm_evaluation_template": self.evaluator.llm_evaluation_template,
                "enable_artifacts": self.evaluator.enable_artifacts,
                "max_artifact_storage": self.evaluator.max_artifact_storage,
                "artifact_keys": self.evaluator.artifact_keys,
                "use_sandboxed_execution": self.evaluator.use_sandboxed_execution,
                "temp_file_suffix": self.evaluator.temp_file_suffix,
                "json_extract_patterns": self.evaluator.json_extract_patterns,
                "sandbox": {
                    "enabled": self.evaluator.sandbox.enabled,
                    "timeout": self.evaluator.sandbox.timeout,
                    "executor_idle_timeout": self.evaluator.sandbox.executor_idle_timeout,
                    "cpu_limit": self.evaluator.sandbox.cpu_limit,
                    "memory_limit_mb": self.evaluator.sandbox.memory_limit_mb,
                    "block_network": self.evaluator.sandbox.block_network,
                    "working_directory": self.evaluator.sandbox.working_directory,
                    "evaluation_volume_path": self.evaluator.sandbox.evaluation_volume_path,
                    "default_dependencies": self.evaluator.sandbox.default_dependencies,
                    "max_concurrent_evaluations": self.evaluator.sandbox.max_concurrent_evaluations,
                    "target_concurrent_evaluations": self.evaluator.sandbox.target_concurrent_evaluations,
                },
            },
            "modal": {
                "enabled": self.modal.enabled,
                "app_name": self.modal.app_name,
                "hub_timeout": self.modal.hub_timeout,
                "hub_max_containers": self.modal.hub_max_containers,
                "hub_max_concurrent_requests": self.modal.hub_max_concurrent_requests,
                "worker_min_containers": self.modal.worker_min_containers,
                "worker_max_containers": self.modal.worker_max_containers,
                "worker_timeout": self.modal.worker_timeout,
                "worker_buffer_size": self.modal.worker_buffer_size,
                "llm_min_containers": self.modal.llm_min_containers,
                "llm_max_containers": self.modal.llm_max_containers,
                "llm_timeout": self.modal.llm_timeout,
                "eval_min_containers": self.modal.eval_min_containers,
                "eval_max_containers": self.modal.eval_max_containers,
                "eval_timeout": self.modal.eval_timeout,
                "eval_retries": self.modal.eval_retries,
                "eval_max_concurrent_inputs": self.modal.eval_max_concurrent_inputs,
                "eval_target_concurrent_inputs": self.modal.eval_target_concurrent_inputs,
                "database_volume_path": self.modal.database_volume_path,
                "evaluation_volume_path": self.modal.evaluation_volume_path,
                "checkpoint_interval_generations": self.modal.checkpoint_interval_generations,
                "export_best_interval_generations": self.modal.export_best_interval_generations,
                "inference_secret_name": self.modal.inference_secret_name,
                "resume_from_checkpoint": self.modal.resume_from_checkpoint,
                "status_poll_interval": self.modal.status_poll_interval,
            },
            # Evolution settings
            "diff_based_evolution": self.diff_based_evolution,
            "max_code_length": self.max_code_length,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
