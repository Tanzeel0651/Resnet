import os

def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)

def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)
        

classes = [
  'Lemon',
  'Mango',
  'Raspberry',
  'Banana',
  'Chestnut',
  'Pear'
]

train_path_from = os.path.abspath('large_dataset/Training')
valid_path_from = os.path.abspath('large_dataset/Validation')

train_path_to = os.path.abspath('small_dataset/Training')
valid_path_to = os.path.abspath('small_dataset/Validation')

mkdir(train_path_to)
mkdir(valid_path_to)

for c in classes:
    link(train_path_from+'/'+c, train_path_to+'/'+c)
    link(valid_path_from+'/'+c, valid_path_to+'/'+c)