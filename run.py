import logging
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import MyData, custom_collate_fn
from model.model import UMEPP as Model
from utils.metrics import compute_metrics
from utils.parsers import build_parser
from utils.optim import get_optim

BLUE = '\033[94m'
ENDC = '\033[0m'
warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):
    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")
    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")
    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")
    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")
    logger.info(BLUE + "Missing Type: " + ENDC + f"{args.available_type}")
    logger.info(BLUE + "Loss Ratio: " + ENDC + f"{args.loss_ratio}")
    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")
    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")
    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")
    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")
    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"train_{args.dataset_id}_{args.available_type}_{timestamp}"
    father_folder_name = args.save_path

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}")

    for i in range(len(model_name_list)):
        if model_name_list[i] != f'checkpoint_{min_turn}_epoch.pkl' and model_name_list[i] != 'log.txt':
            os.remove(os.path.join(f'{father_folder_name}/{folder_name}', model_name_list[i]))


def delete_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace(BLUE, '')
    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def train_val_test(args):
    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)
    df_train = pd.read_pickle(os.path.join(args.dataset_path, 'train.pkl'))
    df_valid = pd.read_pickle(os.path.join(args.dataset_path, 'valid.pkl'))

    train_data = MyData(dataframe=df_train)

    valid_data = MyData(dataframe=df_valid)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=args.batch_size,
                                   collate_fn=custom_collate_fn,
                                   num_workers=args.num_workers)

    valid_data_loader = DataLoader(dataset=valid_data,
                                   batch_size=args.batch_size,
                                   collate_fn=custom_collate_fn,
                                   num_workers=args.num_workers)

    model = Model(
        model_id=args.model_id,
        available_type=args.available_type,
        v_dim=args.v_dim,
        t_dim=args.t_dim,
        a_dim=args.a_dim,
        h_dim=args.h_dim,
    )

    model = model.to(args.device)
    max_steps = len(train_data_loader) * args.epochs
    optim,scheduler = get_optim(max_steps=max_steps,
                                optim_name=args.optim,
                                model=model,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                warmup_rate=args.warmup_rate)
    min_total_valid_loss = 1008611
    min_turn = 0
    init_turn = 0

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        init_turn = checkpoint['epoch']
        min_turn = init_turn
        logger.info(f"Load checkpoint from {args.checkpoint_path} successfully!")
        logger.info(f"Start from {init_turn + 1} epoch!")

    print_init_msg(logger, args)

    for i in range(args.epochs - init_turn):
        logger.info(
            f"-----------------------------------Epoch {i + 1 + init_turn} Start!-----------------------------------")

        min_train_loss, total_valid_loss = run_one_epoch(model=model,
                                                         optim=optim,
                                                         scheduler=scheduler,
                                                         train_data_loader=train_data_loader,
                                                         valid_data_loader=valid_data_loader,
                                                         loss_ratio=args.loss_ratio,
                                                         device=args.device
                                                         )

        logger.info(f"[ Epoch {i + 1 + init_turn} (train) ]: loss = {min_train_loss}")
        logger.info(f"[ Epoch {i + 1 + init_turn} (valid) ]: total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:
            min_total_valid_loss = total_valid_loss
            min_turn = i + 1 + init_turn

        logger.critical(
            f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")

        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optim.state_dict(),
                      "epoch": i + 1, }

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        path_checkpoint = f"{father_folder_name}/{folder_name}/checkpoint_{i + 1 + init_turn}_epoch.pkl"
        torch.save(checkpoint, path_checkpoint)
        logger.info("Model has been saved successfully!")
        delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")

        if (i + 1) - min_turn > args.early_stop_turns:
            break

    delete_model(father_folder_name, folder_name, min_turn)
    logger.info(BLUE + "Training is ended!" + ENDC)
    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")
    checkpoint = torch.load(f"{father_folder_name}/{folder_name}/checkpoint_{min_turn}_epoch.pkl",
                            map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_data = MyData(dataframe=pd.read_pickle(os.path.join(args.dataset_path, 'test.pkl')))
    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=args.test_batch_size,
                                  collate_fn=custom_collate_fn,
                                  num_workers=args.num_workers)
    test_steps = 0
    total_MAE = 0
    total_SRC = 0
    total_nMSE = 0
    total_PCC = 0
   
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc='Testing Progress'):
            batch = [item.to(args.device) if isinstance(item, torch.Tensor) else item for item in batch]
            v_f_seq, t_f, a_f, label, _ = batch
            output, loss_kl, loss_u = model(v_f_seq=v_f_seq, t_f=t_f, a_f=a_f,training=False)
            MAE, SRC, nMSE, PCC = compute_metrics(output, label)
            
            total_MAE += MAE
            total_SRC += SRC
            total_nMSE += nMSE
            total_PCC += PCC
            test_steps += 1
  
    
    test_MAE = total_MAE / test_steps
    test_SRC = total_SRC / test_steps
    test_nMSE = total_nMSE / test_steps
    test_PCC = total_PCC / test_steps
    logger.info(f"Test MAE: {test_MAE}, Test SRC: {test_SRC}, Test nMSE: {test_nMSE}, Test PCC: {test_PCC}")
    logger.info("Finished Testing!")


def run_one_epoch(model, optim, scheduler, train_data_loader, valid_data_loader, loss_ratio, device):
    model.train()
    min_train_loss = 1008611
    for batch in tqdm(train_data_loader, desc='Training Progress'):
        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
        v_f_seq, t_f, a_f, label, item_id = batch
        loss = model.compute_loss(
            v_f_seq=v_f_seq,
            t_f=t_f,
            a_f=a_f,
            label=label,
            loss_ratio=loss_ratio,
            training=True
        )

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        if min_train_loss > loss:
            min_train_loss = loss

    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_data_loader, desc='Validating Progress'):
            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            v_f_seq, t_f, a_f, label, _ = batch
            loss = model.compute_loss(
                v_f_seq=v_f_seq,
                t_f=t_f,
                label=label,
                a_f=a_f,
                loss_ratio=loss_ratio,
                training=False
            )
            total_valid_loss += loss
    return min_train_loss, total_valid_loss


def main():
    parser = build_parser('UMEPP')
    args = parser.parse_args()
    seed_init(args.seed)
    train_val_test(args)


if __name__ == '__main__':
    main()
