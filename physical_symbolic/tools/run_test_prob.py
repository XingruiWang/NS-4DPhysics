import os
import json
import sys
sys.path.append('.')
from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
import utils.utils as utils


def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type

def find_superclevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    elif out_mod.startswith('partquery'):
        q_type = 'partquery'
    return q_type

def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i+1]:
            return False
        if pred[i] == 2:
            break
    return True


opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
executor = get_executor(opt)
# model = Seq2seqParser(opt)

print('| running test')
stats = {
    'error': 0,
    'count': 0,
    'count_tot': 0,
    'exist': 0,
    'exist_tot': 0,
    'compare_num': 0,
    'compare_num_tot': 0,
    'compare_attr': 0,
    'compare_attr_tot': 0,
    'query': 0,
    'query_tot': 0,
    'partquery_tot': 0,
    'correct_ans': 0,
    'correct_ans_no_guess': 0,
    'correct_prog': 0,
    'total': 0
}
with open(opt.superclevr_gt_question_path) as f:
    q_raw = json.load(f)['questions']
            
for x, y, ans, idx, q_idx in loader:

    # model.set_input(x, y)
    # pred_program = model.parse()
    pred_program = y
    # y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()
    y_np, pg_np, idx_np, ans_np = y.numpy(), pred_program.numpy(), idx.numpy(), ans.numpy()
    
    for i in range(y.shape[0]):
        
        pred_ans, is_guess = executor.run(pg_np[i], idx_np[i], 'val')
        # pred_ans = executor.run(pg_np[i], idx_np[i], 'val', guess=True)
        gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]
        
        # print(pred_ans, gt_ans)
        q_type = find_superclevr_question_type(executor.vocab['program_idx_to_token'][pg_np[i][1]])
        # if pred_ans != gt_ans :
        #     print("GT", q_raw[q_idx[i]])
            
        #     pred_ans = executor.run(pg_np[i], idx_np[i], 'val', guess=False, replay = True)

        if is_guess == True:
            stats['error'] += 1
        if pred_ans == gt_ans:
            stats[q_type] += 1
            stats['correct_ans'] += 1
            if not is_guess:
                stats['correct_ans_no_guess'] += 1
        if check_program(pg_np[i], y_np[i]):
            stats['correct_prog'] += 1

        stats['%s_tot' % q_type] += 1
        stats['total'] += 1

    print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))

result = {
    'overall_acc': stats['correct_ans'] / stats['total'],
    'overall_acc_no_guess': stats['correct_ans_no_guess'] / stats['total'],
    'error': stats['error'] / stats['total'],
    'count_acc': stats['count'] / stats['count_tot'],
    'exist_acc': stats['exist'] / stats['exist_tot'],
    'compare_num_acc': stats['compare_num'] / stats ['compare_num_tot'],
    'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
    'query_acc': stats['query'] / stats['query_tot'],
    'program_acc': stats['correct_prog'] / stats['total']


}
print(result)

utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    