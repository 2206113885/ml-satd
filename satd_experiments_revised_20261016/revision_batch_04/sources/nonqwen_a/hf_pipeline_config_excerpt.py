# Source: https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/pipelines/base.py
# L999         # 1 - create a local generation config. This is done to avoid side-effects on the model as we apply local
# L1000         # tweaks to the generation config.
# L1001         # 2 - load the assistant model if it is passed.
# L1002         self.assistant_model, self.assistant_tokenizer = load_assistant_model(
# L1003             self.model, kwargs.pop("assistant_model", None), kwargs.pop("assistant_tokenizer", None)
# L1004         )
# L1005         if self.model.can_generate():
# L1006             self.prefix = self.model.config.prefix if hasattr(self.model.config, "prefix") else None
# L1007             self.generation_config = copy.deepcopy(self.model.generation_config)
# L1008             # Update the generation config with task specific params if they exist
# L1009             # NOTE: `prefix` is pipeline-specific and doesn't exist in the generation config.
# L1010             task_specific_params = self.model.config.task_specific_params
# L1011             if task_specific_params is not None and task in task_specific_params:
# L1012                 this_task_params = task_specific_params.get(task)
# L1013                 if "prefix" in this_task_params:
# L1014                     self.prefix = this_task_params.pop("prefix")
# L1015                 self.generation_config.update(**this_task_params)
# L1016             # If the tokenizer has a pad token but the model doesn't, set it so that `generate` is aware of it.
# L1017             if (
# L1018                 self.tokenizer is not None
# L1019                 and self.tokenizer.pad_token_id is not None
# L1020                 and self.generation_config.pad_token_id is None
# L1021             ):
# L1022                 self.generation_config.pad_token_id = self.tokenizer.pad_token_id
# L1023 
# L1024         self.call_count = 0
# L1025         self._batch_size = kwargs.pop("batch_size", None)
# L1026         self._num_workers = kwargs.pop("num_workers", None)
# L1027         self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
# L1028 
# L1029         # In processor only mode, we can get the modality processors from the processor
# L1030         if self.processor is not None and all(
# L1031             [self.tokenizer is None, self.feature_extractor is None, self.image_processor is None]
# L1032         ):
