#!/usr/bin/env python
# -*- coding=utf-8 -*-
import sys, json
from datetime import datetime, timedelta
import os.path as path
root_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(root_path)
import requests, time, base64, hmac
from configparser import ConfigParser



###################################################
# 获取配置文件
###################################################
def get_config_info(conf_name=''):
    """
    Args:
        conf_name: <str> the name of the config file of the different biz lines
    Returns:
        config object
    """
    # env_list = ['dev', 'prod', 'test']
    # assert env in env_list, ValueError("你传入了一个无效的环境（开发：dwv、生产：prod、测试：test）")
    assert conf_name is not None or len(conf_name) > 3, ValueError("请传入配置文件")
    assert len(conf_name.split(path.sep)) < 2, ValueError("只需要传入配置文件就行，不需要传文件路径")

    config_path = path.join(root_path, 'database_config', conf_name)

    if conf_name.endswith(".ini") or conf_name.endswith(".conf"):  # 读取 .ini  .conf 配置文件
        ini_config = ConfigParser()
        ini_config.read(config_path, encoding='utf8')
        return ini_config

    elif conf_name.endswith(".json"):  # 读取 json 类型的配置文件
        with open(config_path) as json_file:
            json_config = json.load(json_file)
        return json_config
    else:
        return None


























