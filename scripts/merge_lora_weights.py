import dataclasses
import functools
import logging
import platform
from typing import Any
from flax.core import freeze, unfreeze
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import math
import ml_collections

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.models.pi0 as pi0
import openpi.models.lora as lora


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _clean_graph_def(graph_def):
    """Recursively remove keys containing lora suffixes from a nested dict/list/tuple structure."""
    if isinstance(graph_def, dict):
        cleaned = {}
        for k, v in graph_def.items():
            # Check the key itself and potentially nested structures if key represents a module type
            if not any(lora_type in str(k).lower() for lora_type in ["lora_a", "lora_b", "lora_config"]):
                 cleaned[k] = _clean_graph_def(v)
        return cleaned
    elif isinstance(graph_def, (list, tuple)):
        # Filter out list/tuple elements that might be LoRA parameter names directly (less common in graphdefs)
        cleaned_list = []
        for x in graph_def:
             if not (isinstance(x, str) and any(lora_type in x.lower() for lora_type in ["lora_a", "lora_b", "lora_config"])):
                 cleaned_list.append(_clean_graph_def(x))
        return type(graph_def)(cleaned_list)
    else:
        # Base case: return non-dict/list/tuple elements as is
        return graph_def


def convert_to_base_model(train_state: training_utils.TrainState, config: _config.TrainConfig):
    """Convert a LoRA-enabled model state to a base model state by merging weights."""
    logging.info("Starting transform LoRA model to Base model...")

    # 1. Flatten the parameter state first to check for LoRA presence
    params_dict = train_state.params.to_pure_dict()
    flattened_params = traverse_util.flatten_dict(params_dict, keep_empty_nodes=True)

    # 2. Detect if LoRA parameters exist in the state
    found_lora_in_state = False
    for path in flattened_params:
        param_name = path[-1]
        if any(lora_suffix in param_name for lora_suffix in ["lora_a", "lora_b", "_lora_a", "_lora_b"]):
            found_lora_in_state = True
            logging.info(f"Detected LoRA parameter in state: {'/'.join(str(p) for p in path)}")
            break # Found one, no need to check further

    lora_config = None
    # 3. Determine LoRA config: prioritize config file, fallback to hardcoded defaults if state has LoRA
    try:
        if hasattr(config.model, "lora_configs") and config.model.lora_configs:
            logging.info("Found lora_configs in the provided model config.")
            lora_configs_dict = config.model.lora_configs
            if isinstance(lora_configs_dict, ml_collections.ConfigDict):
                lora_configs_dict = lora_configs_dict.to_dict() # Convert if it's a ConfigDict

            if not isinstance(lora_configs_dict, dict) or not lora_configs_dict:
                raise ValueError("config.model.lora_configs is not a valid, non-empty dictionary.")

            # Retrieve one LoRA config (prioritize attn/ffn)
            if 'attn' in lora_configs_dict:
                lora_config_data = lora_configs_dict['attn']
            elif 'ffn' in lora_configs_dict:
                lora_config_data = lora_configs_dict['ffn']
            else:
                first_key = next(iter(lora_configs_dict))
                lora_config_data = lora_configs_dict[first_key]
                logging.warning(f"Could not find 'attn' or 'ffn' in lora_configs. Using config from key '{first_key}'.")

            # Reconstruct LoRAConfig object
            if isinstance(lora_config_data, lora.LoRAConfig):
                lora_config = lora_config_data
            else:
                lora_config = lora.LoRAConfig(**lora_config_data)
        elif found_lora_in_state:
             # Config missing, but LoRA found in state -> Use hardcoded defaults
             default_rank = 16
             default_alpha = 16.0
             default_rslora = False # Assume False unless specified otherwise
             logging.warning("LoRA parameters detected in state, but no lora_configs found in the provided config.")
             logging.warning(f"Using HARDCODED default LoRA parameters: rank={default_rank}, alpha={default_alpha}, rslora={default_rslora}")
             lora_config = lora.LoRAConfig(rank=default_rank, alpha=default_alpha, rslora=default_rslora)

    except Exception as e:
        logging.error(f"Error processing LoRA configuration: {e}")
        if found_lora_in_state:
             logging.error("Cannot proceed with merge due to configuration error.")
             # Optionally, re-raise or return original state. Here we prevent merge.
             lora_config = None # Ensure merge doesn't happen
        # If no LoRA in state anyway, the error might be less critical, let it proceed to cleanup.


    # 4. Proceed with merge or cleanup
    if lora_config:
        scaling_value = lora_config.scaling_value
        logging.info(f"Using LoRA scaling value: {scaling_value} (derived from rank={lora_config.rank}, alpha={lora_config.alpha}, rslora={lora_config.rslora})")

        converted_params = {}
        processed_lora_paths = set() # Keep track of paths already processed

        logging.info("Identifying LoRA layers and corresponding base weights...")
        # Group parameters by their parent module path
        grouped_params = {}
        for path, value in flattened_params.items():
            parent_path = path[:-1]
            param_name = path[-1]
            if parent_path not in grouped_params:
                grouped_params[parent_path] = {}
            grouped_params[parent_path][param_name] = value

        # Iterate through modules to find and merge LoRA weights
        for parent_path, params in grouped_params.items():

            # Check for and merge Einsum LoRA pattern ('w', 'lora_a', 'lora_b')
            if "lora_a" in params and "lora_b" in params:
                if "w" in params:
                    lora_a_key, lora_b_key, base_key = "lora_a", "lora_b", "w"
                    _perform_merge(
                         parent_path, params, lora_a_key, lora_b_key, base_key,
                         scaling_value, converted_params, processed_lora_paths
                    )
                else:
                    path_str_a = "/".join(str(p) for p in parent_path + ("lora_a",))
                    logging.warning(f"Found LoRA parameters {path_str_a} and ..._lora_b, but corresponding base weight 'w' is missing. Skipping merge.")
                    processed_lora_paths.add(parent_path + ("lora_a",))
                    processed_lora_paths.add(parent_path + ("lora_b",))

            # Check for and merge FeedForward gating LoRA pattern
            if "gating_einsum_lora_a" in params and "gating_einsum_lora_b" in params:
                if "gating_einsum" in params:
                    lora_a_key = "gating_einsum_lora_a"
                    lora_b_key = "gating_einsum_lora_b"
                    base_key = "gating_einsum"
                    _perform_merge(
                         parent_path, params, lora_a_key, lora_b_key, base_key,
                         scaling_value, converted_params, processed_lora_paths
                    )
                else:
                    path_str_a = "/".join(str(p) for p in parent_path + ("gating_einsum_lora_a",))
                    logging.warning(f"Found LoRA parameters {path_str_a} and ..._lora_b, but corresponding base 'gating_einsum' is missing. Skipping merge.")
                    processed_lora_paths.add(parent_path + ("gating_einsum_lora_a",))
                    processed_lora_paths.add(parent_path + ("gating_einsum_lora_b",))

            # Check for and merge FeedForward linear LoRA pattern
            if "linear_lora_a" in params and "linear_lora_b" in params:
                if "linear" in params:
                    lora_a_key, lora_b_key, base_key = "linear_lora_a", "linear_lora_b", "linear"
                    _perform_merge(
                         parent_path, params, lora_a_key, lora_b_key, base_key,
                         scaling_value, converted_params, processed_lora_paths
                    )
                else:
                    path_str_a = "/".join(str(p) for p in parent_path + ("linear_lora_a",))
                    logging.warning(f"Found LoRA parameters {path_str_a} and ..._lora_b, but corresponding base weight 'linear' is missing. Skipping merge.")
                    processed_lora_paths.add(parent_path + ("linear_lora_a",))
                    processed_lora_paths.add(parent_path + ("linear_lora_b",))


        # Copy over non-LoRA weights that weren't part of a merge
        logging.info("Copying remaining non-LoRA parameters...")
        for path, value in flattened_params.items():
            if path not in processed_lora_paths:
                 # Check if the path looks like a LoRA parameter that *should* have been processed
                 param_name = path[-1]
                 is_potential_lora = any(lora_suffix in param_name for lora_suffix in ["lora_a", "lora_b", "_lora_a", "_lora_b"])

                 if is_potential_lora:
                     # This should ideally not happen if the checks above were comprehensive.
                     # Log a warning if we find an unprocessed LoRA parameter here.
                     logging.warning(f"Found unprocessed LoRA parameter: {'/'.join(str(p) for p in path)}. This might indicate an issue in the merge logic or unexpected parameter structure.")
                 else:
                     # Copy genuine non-LoRA parameters
                     converted_params[path] = value


        # Convert back to nested dict and then to nnx.State
        converted_dict = traverse_util.unflatten_dict(converted_params)
        final_params = nnx.State(converted_dict)

        # Generate a clean graph definition from a model without LoRA config
        logging.info("Generating clean graph definition from non-LoRA model config...")
        try:
            # Create a clean model config by removing LoRA settings
            clean_model_config = dataclasses.replace(config.model, lora_configs=None)
            # Instantiate a dummy model with this clean config to get its graphdef
            # We need a dummy key for model initialization
            dummy_rng = jax.random.key(0) # Use a fixed arbitrary key
            clean_model = clean_model_config.create(dummy_rng)
            clean_graphdef = nnx.graphdef(clean_model)
            logging.info("Successfully generated clean graph definition.")
        except Exception as e:
            logging.error(f"Failed to generate clean graph definition: {e}")
            logging.warning("Falling back to cleaning the existing graph definition.")
            # Fallback to the previous cleaning method if instantiation fails
            clean_graphdef = _clean_graph_def(train_state.model_def)


        # Create the new train state with merged params and clean graph def
        new_train_state = dataclasses.replace(
            train_state,
            params=final_params,
            model_def=clean_graphdef
        )
        logging.info(f"Final merged state:\n{training_utils.array_tree_to_info(new_train_state.params)}")
        return new_train_state

    else: # No LoRA config found and no LoRA parameters detected in state
        logging.warning("No LoRA config found in model config and no LoRA parameters detected in state. Assuming no merge needed.")
        # Attempt to clean the graph def anyway, in case it contains orphaned LoRA nodes
        cleaned_def = _clean_graph_def(train_state.model_def)
        return dataclasses.replace(train_state, model_def=cleaned_def)


# Define the merge logic as a helper function to avoid repetition
def _perform_merge(
    parent_path: tuple,
    params: dict,
    lora_a_key: str,
    lora_b_key: str,
    base_key: str,
    scaling_value: float,
    converted_params: dict,
    processed_lora_paths: set
):
    """Performs the actual LoRA weight merge calculation."""
    lora_a = params[lora_a_key]
    lora_b = params[lora_b_key]
    base_weight = params[base_key]
    full_base_path = parent_path + (base_key,)
    full_lora_a_path = parent_path + (lora_a_key,)
    full_lora_b_path = parent_path + (lora_b_key,)

    # Avoid re-processing if somehow this combination was already handled (defensive check)
    if full_base_path in processed_lora_paths:
        return

    path_str = "/".join(str(p) for p in full_base_path)
    logging.info(f"Processing LoRA merge for: {path_str}")
    logging.info(f"  Base shape: {base_weight.shape}, dtype: {base_weight.dtype}")
    logging.info(f"  LoRA A shape: {lora_a.shape}, dtype: {lora_a.dtype}")
    logging.info(f"  LoRA B shape: {lora_b.shape}, dtype: {lora_b.dtype}")

    # Cast LoRA weights to the base weight dtype (e.g., bfloat16) *before* matmul
    # to mimic the casting done in the original forward pass (w_a.astype(x.dtype)).
    lora_a_casted = lora_a.astype(base_weight.dtype)
    lora_b_casted = lora_b.astype(base_weight.dtype)
    logging.info(f"  Casting LoRA A/B to {base_weight.dtype} for matmul.")

    # Calculate LoRA delta using the casted dtype
    lora_delta = jnp.matmul(lora_a_casted, lora_b_casted)

    # Apply scaling factor ONLY for Einsum layers (base_key == "w"), NOT for FeedForward layers.
    # Note: Since scaling is 1.0, this step is currently a NOP, but we keep the logic.
    apply_scaling = (base_key == "w")
    if apply_scaling:
        scaled_lora_delta = lora_delta # Effectively: (lora_delta * 1.0).astype(base_weight.dtype)
        logging.info(f"  Applying scaling factor ({scaling_value}) for Einsum merge (effectively NOP as scale=1.0).")
    else:
        scaled_lora_delta = lora_delta
        logging.info(f"  Skipping scaling factor for FeedForward ({base_key}) merge.")

    # Perform the merge. All tensors should now be in base_weight.dtype.
    merged_weight = base_weight + scaled_lora_delta

    logging.info(f"  Merged shape: {merged_weight.shape}, dtype: {merged_weight.dtype}")

    # Store merged weight and mark LoRA paths as processed
    converted_params[full_base_path] = merged_weight
    processed_lora_paths.add(full_lora_a_path)
    processed_lora_paths.add(full_lora_b_path)
    processed_lora_paths.add(full_base_path) # Mark base as processed too, as it's now the merged one

    # Clean up memory
    del lora_a, lora_b, base_weight, lora_a_casted, lora_b_casted, merged_weight, lora_delta, scaled_lora_delta
    jax.clear_caches()


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    step = int(train_state.step)

    # transform model
    train_state = convert_to_base_model(train_state, config)
    logging.info("Transformation Completed")

    # save model, index=step+1 - Use a distinct step or name convention for merged models
    merged_step = step + 1 # Or use a large number, or a specific name
    logging.info(f"Saving merged model state at step {merged_step}...")
    _checkpoints.save_state(checkpoint_manager, train_state, data_loader, merged_step) # Save with new step/ID

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())