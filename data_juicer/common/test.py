from mysql_helper import get_startrocks_helper, get_db_helper



# 获取新增车天，组bundle列表
starrocks_helper = get_startrocks_helper(database='perfect_travel')
meta_qur_sql="""
select 
    annotation_type,sensor_name,bundle_path,annotation_path,demand_source
from perfect_travel.ods_annotation_day
where annotation_type='icu30_2d_3d_key_point_clip'
    and bundle_path != ""
    and sensor_name = "front_middle_camera"
limit 1000;

"""
new_tags_df = starrocks_helper.read_db_to_df(meta_qur_sql)
starrocks_helper.close()
new_tags_df
