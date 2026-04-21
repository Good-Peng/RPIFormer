import importlib
from os import path as osp

from basicsr.utils import get_root_logger, scandir
"""
    动态导入并构建模型实例
    自动扫描并导入 basicsr/models/ 目录下所有以 _model.py 结尾的文件。
    根据配置中的 model_type 名称，在导入的模块中查找对应模型类，并实例化。
"""

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the model modules
_model_modules = [
    importlib.import_module(f'basicsr.models.{file_name}')
    for file_name in model_filenames
]


def create_model(opt):
    """创建模型。

    参数:
        opt (dict): 配置字典，包含：
            model_type (str): 模型类型名，对应类名。
    """
    model_type = opt['model_type']

    # 动态查找类（在导入的模块列表中遍历查找目标类）
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)   # 获取类对象
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
