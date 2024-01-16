from transformers import BertConfig
from transformers import  AutoConfig
def test():
    config = AutoConfig.from_pretrained(
        "",
        num_labels="",
        label2id=dataset.label_to_id,
        id2label={i: l for l, i in dataset.label_to_id.items()},
        revision=model_args.model_revision,
    )
class PTConfig(BertConfig):
    def __init__(self,**kwargs):
        super(PTConfig, self).__init__(**kwargs)

