# -*- coding:utf-8 -*-
"""
:Date: 2023-02-18 23:58:23
:LastEditTime: 2023-02-19 14:01:40
:Description: 
"""
import numpy as np
import pandas as pd
import faiss
import json
import io
import os
import random
from PIL import Image
import altair as alt
import plotly.graph_objects as go
import torch.nn.functional as F
import streamlit as st
from data_juicer.utils.model_utils import get_model, prepare_model
import extra_streamlit_components as stx
from data_juicer.common.mysql_helper import get_starrocks_helper, get_db_helper
from data_juicer.platform.src.utils.vis import draw_sankey_diagram
from data_juicer.platform.src.utils.vis_common import vis_annotation
import streamlit.components.v1 as components
import sweetviz as sv

annotation_categories = {'物流车-3D-点云目标预测': 'hd_3d_box_1002',
                         '物流车-3D-点云目标检测': 'hd_3d_box_1001',
                         '物流车-3D道路边缘线BEV连续帧': 'hd_lane_edge_3d_bev_clip',
                         '物流车-2D-视觉红绿灯': 'hd_2d_traffic_light_1001',
                         '乘用车-2D&3D-障碍物联合标注连续帧': 'icu30_2d_3d_key_point_clip',
                         '乘用车-2D&3D-障碍物联合标注': 'icu30_2d_3d_box_1001',
                         '乘用车-3D-点云目标预测': 'icu30_3d_box_1002',
                         '乘用车-3D-点云目标检测': 'icu30_3d_box_1001',
                         '4D视觉车道线-BEV-3.0': 'lane_clip_stage_1_3.0',
                         '乘用车-3D道路边缘线BEV连续帧': 'icu30_lane_edge_3d_bev_clip',
                         '乘用车-3D-点云道路边缘线': 'road_edge_line_3d_bundle',
                         '乘用车-2D-视觉红绿灯': 'icu30_2d_traffic_light_1001'
                         }


@st.cache_resource
def load_model():
    model_key = prepare_model(model_type='hf_blip', model_key='Salesforce/blip-itm-base-coco')
    model, processor = get_model(model_key)
    return model, processor

@st.cache_resource
def create_faiss_index(emb_list):
    image_embeddings = np.array(emb_list).astype('float32')
    faiss_index = faiss.IndexFlatL2(image_embeddings.shape[1])
    faiss_index.add(image_embeddings)
    return faiss_index

@st.cache_data
def convert_to_parquet(dataframe):
    buffer = io.BytesIO()
    dataframe.to_parquet(buffer, engine='pyarrow')
    parquet_data = buffer.getvalue()
    return parquet_data

def plot_image_clusters(dataset):
    __dj__image_embedding_2d = np.array(dataset['__dj__image_embedding_2d'])
    df = pd.DataFrame(__dj__image_embedding_2d, columns=['x', 'y'])
    df['image'] = dataset['image']
    df['description'] = dataset['__dj__image_caption']

    marker_chart = alt.Chart(df).mark_circle().encode(
        x=alt.X('x', scale=alt.Scale(type='linear', domain=[df['x'].min() * 0.95, df['x'].max() * 1.05]), axis=alt.Axis(title='X-axis')),
        y=alt.Y('y', scale=alt.Scale(type='linear', domain=[df['y'].min() * 0.95, df['y'].max() * 1.05]), axis=alt.Axis(title='Y-axis')),
        href=('image:N'), 
        tooltip=['image', 'description']
    ).properties(
        width=800,
        height=600,
    ).configure_legend(
        disable=False
    )
    return marker_chart


@st.cache_data
def load_data_from_starrocks(category):
    starrocks_helper = get_starrocks_helper(database='perfect_travel')
    meta_qur_sql=f"""
            select 
                *
            from perfect_travel.ods_annotation_day
            where annotation_type='{category}'
                and bundle_path != ""
                and demand_source = "RD"
                -- and sensor_name = "front_middle_camera"
            limit 1000;
        """
    df = starrocks_helper.read_db_to_df(meta_qur_sql)
    starrocks_helper.close()
    return df

@st.cache_data
def count_from_starrocks(category):
    starrocks_helper = get_starrocks_helper(database='perfect_travel')
    meta_qur_sql=f"""
            SELECT COUNT(*)
            FROM perfect_travel.ods_annotation_day
            WHERE annotation_type = '{category}'
               -- AND bundle_path != "";
        """
    df = starrocks_helper.read_db_to_df(meta_qur_sql)
    starrocks_helper.close()
    return df.iloc[0]['count(*)']

@st.cache_data
def load_tag_names_from_mysql(category):
    starrocks_helper = get_starrocks_helper(database='share_pipeline')
    meta_qur_sql=f"""
            SELECT DISTINCT tag_name
            FROM share_pipeline.ads_custom_annotation_tags
            WHERE annotation_type = '{category}';
        """
    df = starrocks_helper.read_db_to_df(meta_qur_sql)
    tag_names = df.tag_name.tolist()
    starrocks_helper.close()
    return tag_names


@st.cache_data
def load_df_from_mysql(sql_str):
    starrocks_helper = get_starrocks_helper(database='share_pipeline')
    df = starrocks_helper.read_db_to_df(sql_str)
    starrocks_helper.close()
    return df

def download_clean_data(category, clean_tag_names):
    starrocks_helper = get_starrocks_helper(database='share_pipeline')
    sql_clean_str = ''
    for tag_name in clean_tag_names:
        sql_clean_str += f" and not (tag_name = '{tag_name}' and tag_value != 0)"
    meta_qur_sql=f"""
            select 
                annotation_path
            from share_pipeline.ads_custom_annotation_tags
            where annotation_type='{category}'
                and demand_source = "RD"
                {sql_clean_str};
        """
    df = starrocks_helper.read_db_to_df(meta_qur_sql)
    print(df)
    starrocks_helper.close()
    return convert_to_parquet(df)


def write():
    chosen_id = stx.tab_bar(data=[
                    stx.TabBarItemData(id="data_show", title="数据展示", description=""),
                    stx.TabBarItemData(id="data_cleaning", title="数据清洗", description=""),
                    stx.TabBarItemData(id="data_mining", title="数据挖掘", description=""),
                    stx.TabBarItemData(id="data_insights", title="数据洞察", description=""),
                ], default="data_show")

    if chosen_id == 'data_show':
        # 选择数据集
        category = annotation_categories[st.selectbox("选择数据类型", list(annotation_categories.keys()))]
        df = load_data_from_starrocks(category)
        cnt = count_from_starrocks(category)

        st.write(f"共有{cnt}条数据, 展示前1000条数据")
        st.dataframe(df, height=350)

        # 数据集示例可视化
        
        with st.expander('show image', expanded=True):
            with st.spinner('图像绘制中...'):
                if st.button("点击展示随机图像"):
                    annotation_path = random.choice(df['annotation_path'])
                    random_image = vis_annotation(category, annotation_path)
                    st.image(random_image, caption=annotation_path, use_column_width=True)
    
    elif chosen_id == 'data_cleaning':
        # 选择数据集
        category = annotation_categories[st.selectbox("选择数据类型", list(annotation_categories.keys()))]
        cnt = count_from_starrocks(category)
        tag_names = load_tag_names_from_mysql(category)
        clean_tag_names = [_ for _ in tag_names if 'issue' in _]
        if len(clean_tag_names) == 0:
            st.error("该品类数据暂未清洗")
            st.stop()
        
        filter_num = {}
        for tag_name in clean_tag_names:
            sql_str = f"""
                SELECT COUNT(*)
                FROM share_pipeline.ads_custom_annotation_tags
                WHERE annotation_type = '{category}'
                    AND demand_source = "RD"
                    AND tag_name = '{tag_name}'
                    AND tag_value != 0;
            """
            df_select = load_df_from_mysql(sql_str)
            filter_num[tag_name] = df_select.iloc[0]['count(*)']
        

        # 绘制 sankey 图
        value_data, labels = [], ['Original']
        source_data = [0, 0]
        target_data = [1, 2]

        for idx, tag_name in enumerate(clean_tag_names):
            source_data.extend([idx * 2 - 1] * 2)
            target_data.extend([idx * 2 + 1, idx * 2 + 2])
            
            retained_num = cnt - filter_num[tag_name]
            value_data.extend([retained_num, cnt - retained_num])
            labels.extend([f"{tag_name}_retained", f"{tag_name}_discarded"])

        pos_x = [0] + [0.9 / len(clean_tag_names) * (i + 1) for i in range(len(clean_tag_names)) for _ in range(2)]
        pos_y = [0.3] + [0.3, 0.8] * len(clean_tag_names)

        fig = draw_sankey_diagram(source_data, target_data, value_data, labels, pos_x, pos_y)
        st.plotly_chart(fig)

        # 数据导出
        st.subheader('二、清洗数据导出')
        df = load_data_from_starrocks(category)
        df_clean = df.iloc[0:100]
        st.dataframe(df)
        st.download_button('下载parquet格式文件',
                           data=download_clean_data(category, clean_tag_names),
                           file_name=category+'.parquet')
        

        # 数据清洗详情
        st.subheader('三、清洗详情')

        for idx,tag_name in enumerate(clean_tag_names):
            sql_str = f"""
                SELECT COUNT(*)
                FROM share_pipeline.ads_custom_annotation_tags
                WHERE annotation_type = '{category}'
                    AND tag_name = '{tag_name}'
                    AND tag_value != 0;
            """
            df_select = load_df_from_mysql(sql_str)
            discarded_num = df_select.iloc[0]['count(*)']
            st.write(f"{tag_name}: 丢弃{discarded_num}条数据", "展示前100条数据")
            
            sql_query_str = f"""
                SELECT *
                FROM share_pipeline.ads_custom_annotation_tags
                WHERE annotation_type = '{category}'
                    AND tag_name = '{tag_name}'
                    AND tag_value != 0
                limit 100;
            """
            df_discard = load_df_from_mysql(sql_query_str)
            st.dataframe(df)


    elif chosen_id == 'data_mining':
        category = annotation_categories[st.selectbox("选择数据类型", list(annotation_categories.keys()))]
        # faiss_index = create_faiss_index(processed_dataset['__dj__image_embedding'])
        model, processor = load_model()

        # 用户输入文本框
        input_text = st.text_input("请输入搜索文本", 'truck')
        search_button = st.button("开始搜索", type="primary", use_container_width=False)

        if search_button:
            inputs = processor(text=input_text, return_tensors="pt")
            text_output = model.text_encoder(inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True) 
            text_feature = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1).detach().cpu().numpy() 
            text_feature = text_feature[0].astype('float32').tolist()

            starrocks_helper = get_starrocks_helper(database='share_pipeline')
            meta_qur_sql=f"""
                    select ext_tag_value, cosine_similarity(array<float>{text_feature}, ext_vectors) AS similarity 
                    from share_pipeline.ads_custom_annotation_tags
                    where annotation_type='{category}' and tag_name = ''
                    ORDER BY similarity DESC
                        limit 10;
                """
            
            df = starrocks_helper.read_db_to_df(meta_qur_sql)
            starrocks_helper.close()
            if df.empty:
                st.warning("该品类数据暂未入库")
                st.stop()
            for idx, info in enumerate(df.ext_tag_value.to_list()):
                info = json.loads(info)
                image_path = os.path.join('/', info['image_path'])
                caption = info['image_caption']
                content = f'Similarity: {df.iloc[idx]["similarity"]}  \n Image_path: {image_path}  \n Image_caption: {caption}'
                st.image(image_path, caption='', use_column_width=False, width=600)
                st.write(content)

    elif chosen_id == 'data_insights':

        all_options = list(annotation_categories.keys())
        col1, col2, col3 = st.columns(3)

        # 左边栏选择数据集
        with col1:
            selected_dataset_1 = st.selectbox('选择数据集1', all_options)

        # 右边栏选择数据集
        with col2:
            selected_dataset_2 = st.selectbox('选择数据集2', ['None'] + all_options)

        with col3:
            st.write(' ')
            analysis_button = st.button("开始分析数据", type="primary", use_container_width=False)

        # 列选择：annotation_type,sensor_name,bundle_path,annotation_path,demand_source
        df1 = load_data_from_starrocks(annotation_categories[selected_dataset_1])

        if selected_dataset_2 != 'None':
            df2 = load_data_from_starrocks(annotation_categories[selected_dataset_2])

        if analysis_button:

            # st.markdown('<iframe src="http://datacentric.club:3000/" width="1000" height="600"></iframe>', unsafe_allow_html=True)
            # st.map()
            with st.expander('数据集对比分析', expanded=True):
                with st.spinner('Wait for process...'):
                    if selected_dataset_2 == 'None':
                        report = sv.analyze(df1)
                    else:
                        report = sv.compare(df1, df2)
                report.show_html(filepath='./frontend/public/EDA.html', open_browser=False, layout='vertical', scale=1.0)
                components.html(open('./frontend/public/EDA.html').read(), width=1100, height=1200, scrolling=True)