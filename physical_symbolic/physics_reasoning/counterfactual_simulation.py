import os
import json
import sys
sys.path.append('.')
from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
import utils.utils as utils
import ipdb



opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
executor = get_executor(opt)
# model = Seq2seqParser(opt)


with open(opt.superclevr_gt_question_path) as f:
    q_raw = json.load(f)['questions'][:]
            
for x, y, ans, idx, q_idx in loader:

    # model.set_input(x, y)
    # pred_program = model.parse()
    pred_program = y
    # y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()
    y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()

    import ipdb; ipdb.set_trace()
    
  