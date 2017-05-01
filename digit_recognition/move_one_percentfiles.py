import svhn

valid_dir = "/Users/arunkumar/ml/data/svhn/valid"
test_dir = "/Users/arunkumar/ml/data/svhn/test"
train_dir = "/Users/arunkumar/ml/data/svhn/train"

# svhn.move_random_files(test_dir, valid_dir, 100, 'test.')
svhn.move_files_and_rename(valid_dir, test_dir)

