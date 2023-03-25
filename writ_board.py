
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import max_error, precision_score, recall_score, average_precision_score, roc_auc_score
import torch
import numpy as np

def write_tensorboard(writer, phase, epoch, acc, loss, y_hat, y_test, probab, scheduler,  optimizer) :
    
    E_y_hat = []
    E_y_test = []
    E_y_probab = []

    E_recall = []
    E_precision = []
    E_lr = []
    E_auc = []

    writer.add_scalar('Accuracy/'+phase, acc, epoch) # acc
    writer.add_scalar('Loss/'+phase, loss, epoch) # loss 각 페이즈 마다 로스와 정확도를 기록
    
    #print(y_hat)

    if phase == 'val' :
        pre_val = precision_score(y_test, y_hat, average='macro') # avg = macro, 각 클래스의 값을 구한다음, 평균, pos_label = 
        recall_val = recall_score(y_test, y_hat, average='macro') # avg = binary, pos_label = 1 (binary기 때문에)
        if scheduler == 0 :
            cur_lr = 0.0001
            print(0.0001)
        else : 
            cur_lr = scheduler.get_lr()[0]
            print(scheduler.get_lr()[0])

        np_y_test = np.array(y_test)
        #print(y_test)
        np_y_probab = np.array(probab)
        np_y_test_onehot = torch.nn.functional.one_hot(torch.Tensor(np_y_test).to(torch.int64), num_classes = 4)

        #cur_ap = average_precision_score(np_y_test, np_y_probab) # avg = macro
        cur_auc = roc_auc_score(np_y_test_onehot, np_y_probab, average='macro', multi_class='ovo') # avg 는 scikiitlearn의 예제를 따라했음 

        writer.add_scalar('Precision', pre_val, epoch) # precision
        writer.add_scalar('Recall', recall_val, epoch) # recall
        writer.add_scalar('Learning rate', cur_lr, epoch) # learning rate
        #writer.add_scalar('Average precision', cur_ap, epoch) # average precision
        writer.add_scalar('AUC', cur_auc, epoch)
        
        E_y_hat = y_hat
        E_y_test = y_test
        E_y_probab = probab

        E_recall = recall_val
        E_precision = pre_val
        E_lr = cur_lr
        E_auc = cur_auc

    return writer, E_y_hat, E_y_test, E_y_probab, E_recall, E_precision, E_lr, E_auc #훈련 페이즈에는 빈리스트만 생성해서 보내다가 테스트 페이즈에서는 각 값으로 바꿔서 리턴
