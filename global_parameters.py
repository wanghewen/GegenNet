import torch, os, uuid

from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryAccuracy, MulticlassF1Score

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def find_gpus(num_of_cards_needed=4, model_parameter_dict={}):
    """
    Find the GPU which uses least memory. Should set CUDA_VISIBLE_DEVICES such that it uses all GPUs.

    :param num_of_cards_needed:
    :return:
    """
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if (torch.cuda.is_available() and not (model_parameter_dict.get("force_cpu", False)))\
            and (os.environ["CUDA_VISIBLE_DEVICES"] not in ["-1", ""]):
        tmp_file_name = f'.tmp_free_gpus_{uuid.uuid4()}'
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >~/{tmp_file_name}')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser(f'~/{tmp_file_name}'), 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2]))
                                   for idx, x in enumerate(frees)]
        os.remove(os.path.expanduser(f'~/{tmp_file_name}'))
        idx_freeMemory_pair.sort(reverse=True)
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        usingGPUs = [idx_memory_pair[0] for idx_memory_pair in
                     idx_freeMemory_pair[:num_of_cards_needed]]
        # usingGPUs = ','.join(usingGPUs)
        print('using GPUs:', end=' ')
        for pair in idx_freeMemory_pair[:num_of_cards_needed]:
            print(f'{pair[0]} {pair[1] / 1024:.1f}GB')
        accelerator = "gpu"
    else:
        usingGPUs = None
        accelerator = "cpu"
    return usingGPUs, accelerator

auroc = BinaryAUROC()
f1 = BinaryF1Score()
macro_f1 = MulticlassF1Score(2, average='macro')
micro_f1 = MulticlassF1Score(2, average='micro')
accuracy = BinaryAccuracy()

@torch.no_grad()
def evaluate(pred_y, y, mode='val', epoch=0, print_result=False, is_best=False):
    if isinstance(pred_y, torch.Tensor):
        preds = pred_y.cpu()
        # preds = pred_y
    else:
        preds = torch.tensor(pred_y)

    if isinstance(y, torch.Tensor):
        # y = y.cpu()
        y = y.to(preds.device)
    else:
        y = torch.tensor(y).to(preds.device)
    auroc.to(preds.device)
    f1.to(preds.device)
    macro_f1.to(preds.device)
    micro_f1.to(preds.device)
    accuracy.to(preds.device)

    y = torch.clamp(y, min=0)  # Ensure that y values are >= 0
    test_y = y

    # 重置指标
    auroc.reset()
    f1.reset()
    macro_f1.reset()
    micro_f1.reset()
    accuracy.reset()

    # Compute metrics
    auc = auroc(preds, test_y).item()
    preds_binary = (preds >= 0.5).float()
    f1_score_value = f1(preds_binary, test_y).item()
    macro_f1_value = macro_f1(preds_binary, test_y).item()
    micro_f1_value = micro_f1(preds_binary, test_y).item()
    accuracy_value = accuracy(preds_binary, test_y).item()
    pos_ratio = torch.sum(test_y).item() / len(test_y)

    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1': f1_score_value,
        f'{mode}_pos_ratio': pos_ratio,
        f'{mode}_epoch': epoch,
        f'{mode}_macro_f1': macro_f1_value,
        f'{mode}_micro_f1': micro_f1_value,
        f'{mode}_accuracy': accuracy_value,
    }

    if print_result:
        if is_best:
            print("Done! Best Results:")
        if "test_epoch" in res and res["test_epoch"] > 0:
            res["val_epoch"] = res["test_epoch"]
        if "val_epoch" not in res:
            res["val_epoch"] = -1
        print_list = ["val_epoch", "test_auc", "test_f1", "test_macro_f1", "test_micro_f1", "test_accuracy"]
        for i in print_list:
            print(i, res[i], end=" ")
        print()

    return res

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
