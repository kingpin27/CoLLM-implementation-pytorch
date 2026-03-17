# CoLLM implementation

## Dataloader

Plan is to create batches offline and arrange them in jsonl file. So while training our custom sampler just returns consecutive samples as batches.