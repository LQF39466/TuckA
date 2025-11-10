TASK_TO_NUM_LABELS_MAPPING = {
    "cola": 2,
    "mnli": 3,
    "qnli": 2,
    "qqp": 2,
    "mrpc": 2,
    "rte": 2,
    "sst2": 2,
}

TASK_TO_INPUT_NAME_1_MAPPING = {
    "cola": "sentence",
    "mnli": "premise",
    "qnli": "question",
    "qqp": "question1",
    "mrpc": "sentence1",
    "rte": "sentence1",
    "sst2": "sentence",
}

TASK_TO_INPUT_NAME_2_MAPPING = {
    "mnli": "hypothesis",
    "qnli": "sentence",
    "qqp": "question2",
    "mrpc": "sentence2",
    "rte": "sentence2",
}

TASK_TO_REMOVE_COLUMNS_MAPPING = {
    "cola": ["sentence", "idx"],
    "mnli": ["premise", "hypothesis", "idx"],
    "qnli": ["question", "sentence", "idx"],
    "qqp": ["question1", "question2", "idx"],
    "mrpc": ["sentence1", "sentence2", "idx"],
    "rte": ["sentence1", "sentence2", "idx"],
    "sst2": ["sentence", "idx"],
}

TASK_TO_VALIDATION_SET_NAME_MAPPING = {
    "cola": "validation",
    "mnli": "validation_matched",
    "qnli": "validation",
    "qqp": "validation",
    "mrpc": "validation",
    "rte": "validation",
    "sst2": "validation",
}