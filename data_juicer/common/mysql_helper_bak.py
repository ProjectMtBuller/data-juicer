#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os, sys, math
import os.path as path

root_path = path.dirname(path.abspath(__file__))
sys.path.append(root_path)
import pymysql
import pandas as pd
import numpy as np
from retrying import retry
from database_config.get_config import get_config_info



# mysql链接
class MysqlHandler:
    def __init__(self, db_host=None, db_port=None, db_user=None, db_password=None, db_name=None):
        assert db_host is not None and db_port is not None and db_user is not None and db_password is not None and db_name is not None, "Missing mysql connection info"
        db_charset = 'utf8'
        try:
            self.db_connection = pymysql.Connect(
                host=db_host
                ,user=db_user
                ,password=db_password
                ,database=db_name
                ,port=db_port
                ,charset=db_charset
                ,read_timeout=180
            )
        except Exception as e:
            print("initialate db conn error: {}".format(str(e)))
            raise e

    """ 增、删、改
    """

    def execute(self, work_sql):
        self.db_connection.ping(reconnect=True)
        with self.db_connection.cursor() as db_cursor:
            try:
                res = db_cursor.execute(work_sql)
                self.db_connection.commit()
            except Exception as e:
                # 回滚
                self.db_connection.rollback()  # ping()
                raise e
            return res

    """ 批量插入
    """

    def batch_insert(self, insert_sql, datas=[]):
        """ 
            work_sql：必须是占位符sql
            datas：必须是 [(), (), ()]
        """
        self.db_connection.ping(reconnect=True)
        with self.db_connection.cursor() as db_cursor:
            try:
                db_cursor.executemany(insert_sql, datas)
                self.db_connection.commit()
            except Exception as e:
                # 回滚
                self.db_connection.rollback()  # ping()
                raise e

    """ 查询
    """

    @retry(stop_max_attempt_number=3, wait_fixed=1)  # 最多重试3次，重试等待间隔1s
    def search_many(self, qur_sql):
        with self.db_connection.cursor(pymysql.cursors.DictCursor) as db_cursor:
            try:
                db_cursor.execute(qur_sql)
                return db_cursor.fetchall()
            except Exception as e:
                raise e

    """ 查询一个
    """

    @retry(stop_max_attempt_number=3, wait_fixed=1)  # 最多重试3次，重试等待间隔1s
    def search_one(self, qur_sql):
        with self.db_connection.cursor(pymysql.cursors.DictCursor) as db_cursor:
            try:
                db_cursor.execute(qur_sql)
                return db_cursor.fetchone()
            except Exception as e:
                raise e

    """ 查询sql，返回pd
    """

    @retry(stop_max_attempt_number=3, wait_fixed=1)  # 最多重试3次，重试等待间隔1s
    def read_db_to_df(self, qur_sql):
        return pd.read_sql_query(sql=qur_sql, con=self.db_connection)

    """ 批量插入
    """

    def batch_insert_db(self, insert_sql, datas=[]):
        """ 
            work_sql：必须是占位符sql
            datas：必须是 [(), (), ()]
        """
        self.db_connection.ping(reconnect=True)
        with self.db_connection.cursor() as db_cursor:
            try:
                fps_num = db_cursor.executemany(insert_sql, datas)
                self.db_connection.commit()
                return fps_num
            except Exception as e:
                # 回滚
                self.db_connection.rollback()  # ping()
                raise e

    """ df 插入数据库
    """

    def df_insert_db(self, table_name, inset_data_df):
        df_data_list = np.array(inset_data_df).tolist()
        df_cols = inset_data_df.columns
        # 插入的SQL语句
        insert_sql = """
        INSERT into {table_name}({table_columns}) 
        VALUES
        ({insert_value})
        """.format(
            table_name=table_name
            ,table_columns=','.join(df_cols)
            ,insert_value=','.join(['%s'] * len(df_cols))
        )
        self.db_connection.ping(reconnect=True)
        with self.db_connection.cursor() as db_cursor:
            try:
                fps_num = db_cursor.executemany(insert_sql, df_data_list)
                self.db_connection.commit()
                return fps_num
            except Exception as e:
                raise e

    def update(self, table_name, data_dict_list, condition):
        set_str = []

        for k, v in data_dict_list.items():
            if type(v) == str:
                set_str.append(f"{k}='{v}'")
            else:
                set_str.append(f"{k}={v}")
        sql = f'''
            UPDATE {table_name}
            SET {','.join(set_str)}
            WHERE {condition}
        '''
        return self.execute(sql)

    """ 关闭数据库连接
    """

    def close(self):
        self.db_connection.close()





##############################################################
#     doris、mysql连接实例
##############################################################
def get_starrocks_helper(database=""):
    starrocks_config = get_config_info(conf_name='application.conf')
    db_host = starrocks_config.get('starrocks', 'db_host')
    db_port = starrocks_config.getint('starrocks', 'db_port')
    db_username = starrocks_config.get('starrocks', 'db_username')
    db_password = starrocks_config.get('starrocks', 'db_password')

    return MysqlHandler(
        db_host=db_host,
        db_port=db_port,
        db_user=db_username,
        db_password=db_password,
        db_name=database
    )


##############################################################
#     doris、mysql连接实例
##############################################################
def get_db_helper(config_key='', database=""):
    config_info = get_config_info(conf_name='application.conf')
    db_host = config_info.get(config_key, 'db_host')
    db_port = config_info.getint(config_key, 'db_port')
    db_username = config_info.get(config_key, 'db_username')
    db_password = config_info.get(config_key, 'db_password')

    return MysqlHandler(
        db_host=db_host,
        db_port=db_port,
        db_user=db_username,
        db_password=db_password,
        db_name=database
    )
