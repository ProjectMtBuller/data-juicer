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
import random
import datetime


# mysql链接
class MysqlHandler:
    def __init__(self, db_host=None, db_port=None, db_user=None, db_password=None, db_name=None):
        assert db_host is not None and db_port is not None and db_user is not None and db_password is not None and db_name is not None, "Missing mysql connection info"
        db_charset = 'utf8'
        try:
            self.db_connection = pymysql.Connect(
                host=random.choice(str(db_host).split(',')).strip()
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



    """ parquet文件，采用broker_load导入
    """
    # parquet文件，采用broker_load导入
    def parquet_broker_load(self, tos_file, db_name, tb_name, columns=[]):
        assert len(columns)>0, "ERROR: need at least one column to load"
        assert len(tb_name)>3 and len(tos_file)>5, "ERROR: need a valid table name and a valid tos file"
        column_names = str(tuple(columns)).replace("'", "")
        # 账号
        tos_config = get_config_info(conf_name='application.conf')
        access_key_id = tos_config.get('tos_access', 'access_key_id')
        access_key_secret = tos_config.get('tos_access', 'access_key_secret')
        endpoint_s3 = tos_config.get('tos_access', 'endpoint_s3')
        # 加载
        label_name = "{db_name}_{tb_name}_{date_time}".format(db_name=db_name, tb_name=tb_name, date_time=datetime.now().strftime("%Y%m%d%H%M%S_%f"))
        broker_load_sql="""
        LOAD LABEL {db_name}.{label_name}
        (
            DATA INFILE('tos://haomo-generalization/{tos_file}')
            INTO TABLE {tb_name}
            FORMAT AS "parquet"
            {column_names}
        )
        WITH BROKER
        (
            "aws.s3.endpoint" = "{endpoint_s3}",
            "aws.s3.access_key" = "{access_key_id}",
            "aws.s3.secret_key" = "{access_key_secret}"
        )
        PROPERTIES
        (
            "max_filter_ratio"="0.005"
            ,"timeout"="1800"
        );
        """.format(
            # ,"load_mem_limit"="4"
            db_name=db_name
            ,tb_name=tb_name
            ,tos_file=tos_file
            ,column_names=column_names
            ,label_name=label_name
            ,access_key_id=access_key_id
            ,access_key_secret=access_key_secret
            ,endpoint_s3=endpoint_s3
        )
        print("库名：", db_name, "===>表名：", tb_name, "===>label名：", label_name, "===>数据地址：", tos_file, "===>>>时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.execute(broker_load_sql)
        # db_connection.execute(broker_load_sql)
        # 查询导出状态
        # 查看数据库 release_15 中的导入作业，且label含有label_dwd_location，状态为 FINISHED
        """
            FAILED : 没有这个label
            CANCELLED : 导入作业失败
            FINISHED: 导入作业成功
        """
        # 任务状态   
        total_exec_num=1
        task_status=''
        scan_rows=0
        sink_rows=0
        job_details=''
        # 查询任务状态
        load_status_qur_sql="""
        SELECT
            JOB_ID  job_id
            ,LABEL  label_name
            ,STATE  load_status
            ,SCAN_ROWS  scan_rows
            ,SINK_ROWS  sink_rows
            ,CREATE_TIME  create_time
            ,LOAD_FINISH_TIME  done_time
            ,ERROR_MSG  error_msg
            ,JOB_DETAILS  job_details
        FROM information_schema.loads 
        WHERE LABEL='{label_name}';
        """.format(label_name=label_name)
        while task_status not in ['FAILED', 'FINISHED', 'CANCELLED'] and total_exec_num<180:
            time.sleep(10)   # 先等待10s
            load_status_json_info = self.search_one(load_status_qur_sql)
            # load_status_json_info = db_connection.search_one(load_status_qur_sql)
            load_status = load_status_json_info.get('load_status', '')
            task_status = '' if load_status is None else load_status
            print("查询次数==", total_exec_num, "导入状态：", task_status)
            scan_rows = int(load_status_json_info.get('scan_rows', 1))
            sink_rows = int(load_status_json_info.get('sink_rows', 0))
            error_msg = load_status_json_info.get('error_msg', '')
            # job_details = load_status_json_info.get('job_details', '')
            # 更新计数
            total_exec_num = total_exec_num + 1
        # 判断状态
        if task_status in ['FINISHED']:
            print("导入状态：", task_status, "===>>>待load行数：", scan_rows, " ===>>>完成load行数：", sink_rows, " ===>>>成功率：", round(sink_rows*100.0/scan_rows, 2), "===>>>时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return 200
        elif task_status in ['CANCELLED']:
            print("导入状态：", task_status, " 失败日志：", error_msg, "===>>>时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return 500
        else:
            print("导入状态：", task_status, " 失败日志：", error_msg, "===>>>时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return 404







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
