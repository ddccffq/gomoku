import os
import torch
import pickle
import tempfile
import datetime
import shutil

class ModelSaver:
    """ģ�ͱ��渨�����ߣ��ṩ���ֱ��淽���ʹ�����"""
    
    def __init__(self, primary_dir=None, log_func=print):
        """��ʼ�����湤��
        
        Args:
            primary_dir: ��Ҫ����Ŀ¼
            log_func: ��־��¼����
        """
        self.primary_dir = primary_dir
        self.log_func = log_func
        
        # ������ʱ����Ŀ¼
        self.temp_dir = os.path.join(
            tempfile.gettempdir(), 
            f"model_save_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # ȷ��Ŀ¼����
        if primary_dir:
            os.makedirs(primary_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.log_func(f"ModelSaver��ʼ�� - ��Ŀ¼: {primary_dir}, ��ʱĿ¼: {self.temp_dir}")
    
    def save_model(self, model, filename, metadata=None):
        """����ģ�ͣ����Զ��ַ���
        
        Args:
            model: Ҫ�����PyTorchģ��
            filename: �ļ���������·����
            metadata: ��ѡ��Ԫ�����ֵ�
        
        Returns:
            bool: �Ƿ�ɹ�����
            str: �ɹ�������ļ�·�������ʧ����ΪNone
        """
        # ����Ŀ��·��
        primary_path = None
        if self.primary_dir:
            primary_path = os.path.join(self.primary_dir, filename)
        
        temp_path = os.path.join(self.temp_dir, filename)
        
        # ���Ա��淽ʽ1: ��Ŀ¼torch.save
        if primary_path:
            try:
                torch.save(model.state_dict(), primary_path)
                self.log_func(f"? ģ���ѱ��浽��Ŀ¼: {primary_path}")
                
                # ����ļ��Ƿ���Ĵ���
                if os.path.exists(primary_path) and os.path.getsize(primary_path) > 0:
                    return True, primary_path
                else:
                    self.log_func(f"?? �ļ��ѱ��浫��СΪ0�򲻴���: {primary_path}")
            except Exception as e:
                self.log_func(f"? �޷����浽��Ŀ¼: {str(e)}")
        
        # ���Ա��淽ʽ2: ��ʱĿ¼torch.save
        try:
            torch.save(model.state_dict(), temp_path)
            self.log_func(f"? ģ���ѱ��浽��ʱĿ¼: {temp_path}")
            
            # ����ļ�
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # �����Ŀ¼����ʧ�ܵ���ʱĿ¼�ɹ������Ը���
                if primary_path and not os.path.exists(primary_path):
                    try:
                        shutil.copy2(temp_path, primary_path)
                        self.log_func(f"? �Ѵ���ʱĿ¼���Ƶ���Ŀ¼")
                        return True, primary_path
                    except Exception as e:
                        self.log_func(f"?? ���Ƶ���Ŀ¼ʧ��: {str(e)}")
                
                return True, temp_path
        except Exception as e:
            self.log_func(f"? ���浽��ʱĿ¼ʧ��: {str(e)}")
        
        # ���Ա��淽ʽ3: pickle
        pickle_path = temp_path + ".pickle"
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            self.log_func(f"? ģ����ͨ��pickle����: {pickle_path}")
            return True, pickle_path
        except Exception as e:
            self.log_func(f"? pickle����ʧ��: {str(e)}")
        
        # ���з�����ʧ����
        return False, None
    
    def cleanup(self):
        """������ʱ�ļ�"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.log_func(f"��������ʱĿ¼: {self.temp_dir}")
        except Exception as e:
            self.log_func(f"������ʱĿ¼ʧ��: {str(e)}")
