import numpy as np

def loadf_2list(file):
	# load npz as a list #
    container_random = np.load(file)
    return [container_random[key] for key in container_random]

data_key = 'dataset_a'

file_train = 'D:\WeSync\Codes\Proj_NS_1\Data\{}\Train_data\Record_Neu300_input7d_IDDoutput2d_train_trial20000__20190824-042126.npz'. \
            format(data_key)
file_test = 'D:\WeSync\Codes\Proj_NS_1\Data\{}\Test_data\Record_Neu300_input7d_IDDoutput2d_test_trial10000__20190824-050513.npz'. \
            format(data_key)

dataset = {'train': loadf_2list(file_train),
            'test': loadf_2list(file_test)}
np.save('{}.npy'.format(data_key),dataset)

# dataset = np.load('dataset_0.npy').item()
# dataset = np.load('D:\WeSync\Codes\Proj_NS_1\Data\dataset_0.npy').item()


data_key = 'NeuralData_3X_1110'
file_train = 'D:\WeSync\Codes\Proj_NS_1\Data\{}\Train_data\Record_Neu300_input7d_IDDoutput2d_train_trial20000__20190824-042126.npz'. \
            format(data_key)
file_test = 'D:\WeSync\Codes\Proj_NS_1\Data\{}\Test_data\Record_Neu300_input7d_IDDoutput2d_test_trial10000__20190824-050513.npz'. \
            format(data_key)