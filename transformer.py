import time

import numpy as np
from datasets import Dataset
import shutil

# Inspiration: https://github.com/graphcore/examples/blob/master/nlp/bert/tensorflow2/run_squad.py

###############
# CREATE DATA #
###############
seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")





from transformers import TFAutoModel

model = TFAutoModel.from_pretrained("bert-base-uncased")

default_args = {"output_dir": "./bert_output_dir"}
shutil.rmtree(default_args["output_dir"], ignore_errors=True)  # Remove directory
import os
os.makedirs(default_args["output_dir"])

from transformers import TrainingArguments, Trainer, logging

training_args = TrainingArguments(num_train_epochs=2, per_device_train_batch_size=4, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)

st = time.time()
trainer.train()
trianin_time = time.time() - st
