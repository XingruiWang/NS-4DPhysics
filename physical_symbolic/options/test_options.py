from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Test Option Class"""

    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--load_checkpoint_path', type=str, help='checkpoint path')
        self.parser.add_argument('--save_result_path', type=str, help='save result path')
        self.parser.add_argument('--max_val_samples', default=None, type=int, help='max val data')
        self.parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
        self.parser.add_argument('--train_dataset',type=str )
        self.parser.add_argument('--test_dataset', type=str)

        

        self.is_train = False