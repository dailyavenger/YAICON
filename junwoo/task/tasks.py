from registry import registry
from task.base_task import BaseTask

@registry.register_task("graph_text_pretrain")
class GraphTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, cuda_enabled=True):
        pass