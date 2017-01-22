dir = 'data'
if not os.path.exists(dir):
    os.makedirs(dir)
    file_name = './data/data.zip'
    data_url = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip'
    request.urlretrieve(data_url, file_name)
    shutil.unpack_archive(file_name, 'data')


training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']