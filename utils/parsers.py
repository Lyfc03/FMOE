import argparse
import yaml


def load_yaml(path):

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def build_parser(mode):

    parser = argparse.ArgumentParser()

    if mode == 'UMEPP':
        config = load_yaml(r'config/config.yaml')
        parser.add_argument('--seed', type=int, default=config['TRAIN']['SEED'])
        parser.add_argument('--num_workers', type=str, default=config['TRAIN']['NUM_WORKERS'])
        parser.add_argument('--device', type=str, default=config['TRAIN']['DEVICE'])
        parser.add_argument('--loss_ratio', type=str, default=config['TRAIN']['LOSS_RATIO'])
        parser.add_argument('--load_checkpoint', type=bool, default=config['CHECKPOINTS']['LOAD_CHECKPOINT'])
        parser.add_argument('--checkpoint_path', type=str, default=config['CHECKPOINTS']['PATH'])
        parser.add_argument('--batch_size', type=int, default=config['TRAIN']['BATCH_SIZE'])
        parser.add_argument('--epochs', type=int, default=config['TRAIN']['MAX_EPOCH'])
        parser.add_argument('--early_stop_turns', type=int, default=config['TRAIN']['EARLY_STOP_TURNS'])

        parser.add_argument('--test_batch_size', type=int, default=config['TEST']['BATCH_SIZE'])

        parser.add_argument('--model_id', type=str, default=config['MODEL']['MODEL_ID'])
        parser.add_argument('--v_dim', type=int, default=config['MODEL']['V_DIM'],help='The dimension of the visual feature.')
        parser.add_argument('--a_dim', type=int, default=config['MODEL']['A_DIM'],help='The dimension of the visual feature.')
        parser.add_argument('--t_dim', type=int, default=config['MODEL']['T_DIM'],help='The dimension of the textual feature.')
        parser.add_argument('--h_dim', type=int, default=config['MODEL']['H_DIM'],help='The dimension of the hidden layer in reconstruction net.')
        parser.add_argument('--dropout', type=float, default=config['MODEL']['DROPOUT'],help='The dropout rate.')

        parser.add_argument('--optim', type=str, default=config['OPTIM']['NAME'])
        parser.add_argument('--lr', type=float, default=config['OPTIM']['LR'])
        parser.add_argument('--weight_decay', type=float, default=config['OPTIM']['WEIGHT_DECAY'])
        parser.add_argument('--warmup_rate', type=float, default=config['OPTIM']['WARMUP_RATE'])

        parser.add_argument('--available_type',type=str,default=config['DATASET']['CONFIG']['AVAILABLE_TYPE'])
        parser.add_argument('--dataset_path', type=str, default=config['DATASET'][config['TRAIN']['DATASET']]['PATH'])
        parser.add_argument('--dataset_id', type=str, default=config['DATASET'][config['TRAIN']['DATASET']]['DATASET_ID'])

        parser.add_argument('--save_path', type=str, default=config['TRAIN']['SAVE_PATH'])
    return parser
