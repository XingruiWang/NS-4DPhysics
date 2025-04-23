from .superclevr_questions import SuperClevrQuestionDataset
from torch.utils.data import DataLoader


def get_dataset(opt, split):
    """Get function for dataset class"""
    assert split in ['train', 'val']

    if opt.dataset == 'clevr':
        if split == 'train':
            question_h5_path = opt.superclevr_question_path
            max_sample = opt.max_train_samples
        else:
            question_h5_path = opt.superclevr_question_path
            max_sample = opt.max_val_samples
        print(opt.length)
        dataset = SuperClevrQuestionDataset(question_h5_path, max_sample, opt.superclevr_vocab_path, length = opt.length)
    else:
        raise ValueError('Invalid dataset')

    return dataset


def get_dataloader(opt, split):
    """Get function for dataloader class"""
    dataset = get_dataset(opt, split)
    shuffle = opt.shuffle if split == 'train' else 0
    loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers)
    print('| %s %s loader has %d samples' % (opt.dataset, split, len(loader.dataset)))
    return loader