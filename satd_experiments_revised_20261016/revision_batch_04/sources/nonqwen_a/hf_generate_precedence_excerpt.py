# Source: https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/generation/utils.py
# Original lines 1641-1719
    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # parameterization priority:
        # kwargs > non-global default values in `generation_config` > `model.generation_config` > GenerationConfig()
        # TODO (joao): per-model generation config classes.

        using_model_generation_config = False
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # the following conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) there are non-default generation parameters in the model config.
            # 4) the user must have set new generation parameters in the model config.
            if (
                self.generation_config._from_model_config  # 1)
                and self.generation_config._original_object_hash == hash(self.generation_config)  # 2)
                and len(self.config._get_non_default_generation_parameters()) > 0  # 3)
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:  # 4)
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed in v5."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )",
                        UserWarning,
                    )
                    self.generation_config = new_generation_config

            generation_config = self.generation_config
            using_model_generation_config = True

        # `torch.export.export` usually raises an exception if it is called
        # with ``strict=True``. deepcopy can only be processed if ``strict=False``.
        generation_config = copy.deepcopy(generation_config)

        if not using_model_generation_config:
            # If `generation_config` is provided:
            # - `use_model_defaults`: let's fallback ALL default values to the model's generation config
            # - otherwise: legacy behavior, let's just make sure we have the tokens defined
            model_base_version = version.parse(version.parse(self.generation_config.transformers_version).base_version)
            if use_model_defaults is True or (
                use_model_defaults is None and model_base_version >= version.parse("4.50.0")
            ):
                modified_values = {}
                default_generation_config = GenerationConfig()
                for key, default_value in default_generation_config.__dict__.items():
                    if key.startswith("_") or key == "transformers_version":  # metadata
                        continue
                    custom_gen_config_value = getattr(generation_config, key)
                    model_gen_config_value = getattr(self.generation_config, key)
                    if custom_gen_config_value == default_value and model_gen_config_value != default_value:
                        modified_values[key] = model_gen_config_value
                        setattr(generation_config, key, model_gen_config_value)
                if len(modified_values) > 0:
                    logger.warning_once(
                        f"`generation_config` default values have been modified to match model-specific defaults: "
                        f"{modified_values}. If this is not desired, please set these values explicitly."
                    )
            else:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.decoder_start_token_id is None:
                    generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

        # Finally, apply any passed kwargs
        model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs
