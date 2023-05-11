import logging
import transformers
from src.atlas import Atlas
from src.util import set_optim
from src.model_io import _cast_and_set_attrs_and_send_to_device

logger = logging.getLogger(__name__)

# initialize the baseline model from Atlas to make it easier to use with the existing Atlas code
class BaselineModel(Atlas):
    def __init__(self, opt, reader, reader_tokenizer):
        super(Atlas, self).__init__()

        self.reader = reader
        self.reader_tokenizer = reader_tokenizer
        self.retriever = None
        self.retriever_tokenizer = None
        self.opt = opt

        self.READER_ALL_TOKENS = list(self.reader_tokenizer.vocab.values())
        
def load_model(opt, eval_only=False):
    if "t5" in opt.model_path:
        reader = transformers.T5ForConditionalGeneration.from_pretrained(opt.model_path)
        reader_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.model_path)
    elif "bert-base-cased" in opt.model_path:
        # to be implemented
        reader = None
        reader_tokenizer = None
    else:
        raise ValueError("train_baseline only supports training T5 or BERT-base-cased models")

    model = BaselineModel(opt, reader, reader_tokenizer)
    model = _cast_and_set_attrs_and_send_to_device(model, opt)
    logger.info(f"Model loaded from {opt.model_path}")
    
    if eval_only:
        return model, None, None
    
    optimizer, scheduler, _, _ = set_optim(opt, model)
    return model, optimizer, scheduler