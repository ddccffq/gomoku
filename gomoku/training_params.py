# ѵ����������
TRAINING_PARAMS = {
    # ��������
    'learning_rate': 0.001,     # ��ʼѧϰ��
    'weight_decay': 1e-4,       # L2���򻯲���
    'epochs': 100,              # ѵ������(�������ӵ�100������)
    'batch_size': 64,           # ����С(�����ڴ����)
    
    # ѧϰ�ʵ���
    'lr_scheduler': True,       # ����ѧϰ�ʵ���
    'lr_decay_step': 20,        # ÿ20�ֽ���ѧϰ��
    'lr_decay_rate': 0.5,       # ѧϰ��˥����
    
    # ������ǿ
    'data_augmentation': True,  # ������ת/��ת��ǿ
    
    # ���̱��뷽ʽ
    'board_encoding': 'binary', # �����Ʊ��룬���ֺڰ�����
    
    # ѵ������
    'self_play_games': 500,     # ���Ҷ������ɵ���Ϸ��(��������)
    'mcts_simulations': 400,    # ÿ��MCTSģ�����
    'c_puct': 5,                # MCTS̽������
    
    # ģ�ͼܹ�
    'num_residual_blocks': 19,  # �в������
    'num_filters': 256,         # ������˲�������
}
