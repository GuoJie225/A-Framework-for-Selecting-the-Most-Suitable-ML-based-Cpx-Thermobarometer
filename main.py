import os
import numpy as np
import pandas as pd
from io import BytesIO
import streamlit as st
from scipy.spatial.distance import mahalanobis
    
def Cal_Madistance(points, dataset):
    md = np.empty((points.shape[0], 1))

    mean_vec = np.mean(dataset, axis=0)
    cov = np.cov(dataset.T)
    inv_cov = np.linalg.pinv(cov)

    for i in range(points.shape[0]):
        md[i] = mahalanobis(points[i], mean_vec, inv_cov)
    return md

def Cal_Recommendation_score(df, model):
    thresholds = {'P20': {'Cpx-only': 6.759, 'Cpx-Liq': 8.700}, 'H21': {'Cpx-only': 7.133},
                  'J22': {'Cpx-only': 7.032, 'Cpx-Liq': 9.436}, 'C23': {'Cpx-only': 8.216},
                  'AL24': {'Cpx-only': 5.957, 'Cpx-Liq': 7.980}}

    accuracy_dict = {'P20': {'Cpx-only': [1], 'Cpx-Liq': [0, 0]}, 'H21': {'Cpx-only': [0.605, 0.447]},
                     'J22': {'Cpx-only': [0.558, 0], 'Cpx-Liq': [1, 1]}, 'C23': {'Cpx-only': [0, 1]},
                     'AL24': {'Cpx-only': [0.558, 0.085], 'Cpx-Liq': [0.818, 0.978]}}

    if model == 'Cpx-Liq':
        models = ['P20', 'J22', 'AL24']
        for i in range(3):
            col = df.columns[i]
            mod = models[i]

            df[mod + '_Domain'] = np.where(df[col] <= thresholds[mod][model], 'In', 'Out')
            df[mod + '_P_score'] = np.where(df[col] <= thresholds[mod][model],
                                                          accuracy_dict[mod][model][0] * 0.8 + (1-df[col]/thresholds[mod][model]) * 0.2, 0)
            df[mod + '_T_score'] = np.where(df[col] <= thresholds[mod][model],
                                                          accuracy_dict[mod][model][1] * 0.8 + (1-df[col]/thresholds[mod][model]) * 0.2, 0)
        P_cols = [mod + '_P_score' for mod in models]
        T_cols = [mod + '_T_score' for mod in models]

        df['P_Recommendation'] = [i.split('_')[0] for i in df[P_cols].idxmax(axis=1)]
        df['T_Recommendation'] = [i.split('_')[0] for i in df[T_cols].idxmax(axis=1)]

    if model == 'Cpx-only':
        models = ['P20', 'H21', 'J22', 'C23', 'AL24']
        for i in range(5):
            col = df.columns[i]
            mod = models[i]
            df[mod + '_Domain'] = np.where(df[col] <= thresholds[mod][model], 'In', 'Out')
            if mod == 'P20':
                df[mod + '_P_score'] = np.where(df[col] <= thresholds[mod][model],
                                                              accuracy_dict[mod][model][0] * 0.8 + (
                                                                      1 - df[col] / thresholds[mod][model]) * 0.2, 0)
            else:
                df[mod + '_P_score'] = np.where(df[col] <= thresholds[mod][model],
                                                          accuracy_dict[mod][model][0] * 0.8 + (
                                                                      1 - df[col] / thresholds[mod][model]) * 0.2, 0)
                df[mod + '_T_score'] = np.where(df[col] <= thresholds[mod][model],
                                                          accuracy_dict[mod][model][1] * 0.8 + (
                                                                      1 - df[col] / thresholds[mod][model]) * 0.2, 0)
        P_cols = [mod + '_P_score' for mod in models]
        T_cols = [mod + '_T_score' for mod in models]
        T_cols.remove('P20_T_score')

        df['P_Recommendation'] = [i.split('_')[0] for i in df[P_cols].idxmax(axis=1)]
        df['T_Recommendation'] = [i.split('_')[0] for i in df[T_cols].idxmax(axis=1)]

    return df


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(current_dir, 'TrainingDataset', 'P20-dataset.xlsx')
file_path2 = os.path.join(current_dir, 'TrainingDataset', 'H21-dataset.xlsx')
file_path3 = os.path.join(current_dir, 'TrainingDataset', 'J22-dataset.xlsx')
file_path4 = os.path.join(current_dir, 'TrainingDataset', 'C23-dataset.xlsx')
file_path5 = os.path.join(current_dir, 'TrainingDataset', 'AL24-dataset.xlsx')

P20 = pd.read_excel(file_path1).fillna(0.01)
H21 = pd.read_excel(file_path2).fillna(0.01)
J22 = pd.read_excel(file_path3).fillna(0.01)
C23 = pd.read_excel(file_path4).fillna(0.01)
AL24 = pd.read_excel(file_path5).fillna(0.01)

tr_dict = {'P20': {'Cpx-only': P20.iloc[:, 1:11], 'Cpx-Liq': P20.iloc[:, 1:23]}, 'H21': {'Cpx-only': H21.iloc[:, 1:13]},
                  'J22': {'Cpx-only': J22.iloc[:, 1:13], 'Cpx-Liq': J22.iloc[:, 1:25]}, 'C23': {'Cpx-only': C23.iloc[:, 1:13]},
                  'AL24': {'Cpx-only': AL24.iloc[:, 1:10], 'Cpx-Liq': AL24.iloc[:, 1:19]}}

# web
st.set_page_config(
    page_title="ML-based Cpx Thermobarometer",
    layout="wide",  
    initial_sidebar_state="auto"
)

st.title(':blue[A Framework for Selecting the Most Suitable ML-based Cpx Thermobarometer :earth_asia:]')
st.markdown(':blue[Mahalanobis distance] quantitatively characterizes how well a new sample fits the applicability domain of each ML‑based Cpx thermobarometer. Higher prediction accuracy is achieved within the domain. This framework pre‑calculates the distance between your sample and each training dataset, helping you select the most reliable thermobarometer for your petrological application.')

st.header('1. Prepare Your Data')
st.session_state.model = st.radio("Choose your thermobarometer type: ", ["Cpx-Liq", "Cpx-only"])

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data' not in st.session_state:
    st.session_state.data = None


@st.cache_data

def to_template_df(model):
    output = BytesIO()

    input_Cpx_Liq_excel = pd.DataFrame(
        columns=['Sample', 'SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'Cr2O3_Cpx', 'FeOt_Cpx', 'MnO_Cpx', 'MgO_Cpx', 'CaO_Cpx', 'Na2O_Cpx', 'K2O_Cpx', 'NiO_Cpx', 'P2O5_Cpx',
                 'SiO2_Liq', 'TiO2_Liq', 'Al2O3_Liq', 'Cr2O3_Liq', 'FeOt_Liq', 'MnO_Liq', 'MgO_Liq', 'CaO_Liq', 'Na2O_Liq', 'K2O_Liq', 'NiO_Liq', 'P2O5_Liq', 'H2O_Liq'])
    input_Cpx_only_excel = pd.DataFrame(
        columns=['Sample', 'SiO2_Cpx', 'TiO2_Cpx', 'Al2O3_Cpx', 'Cr2O3_Cpx', 'FeOt_Cpx', 'MnO_Cpx', 'MgO_Cpx', 'CaO_Cpx', 'Na2O_Cpx', 'K2O_Cpx', 'NiO_Cpx', 'P2O5_Cpx'])

    if model == "Cpx-Liq":
        df = input_Cpx_Liq_excel
    else:
        df = input_Cpx_only_excel

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    template_df = output.getvalue()
    return template_df


Template_excel = to_template_df(st.session_state.model)

st.download_button(
    label="Download Template",
    data=Template_excel,
    file_name=st.session_state.model + "_Input_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="download_template_button"
)

st.divider()

st.header('2. Upload Your Data')
uploaded_file = st.file_uploader("Analytical data:", type=['xlsx', 'csv'], accept_multiple_files=False)

if uploaded_file is not None:
    if st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        if uploaded_file.name.split('.')[-1] == 'xlsx':
            st.session_state.data = pd.read_excel(uploaded_file).fillna(0.01)
        else:
            st.session_state.data = pd.read_csv(uploaded_file).fillna(0.01)
        st.session_state.mahalanobis_distance = False

if st.session_state.data is not None:
    st.dataframe(st.session_state.data)

st.divider()

st.header('3. Calculate Mahalanobis Distance')


@st.cache_data
def to_result_df(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    download_file = output.getvalue()
    return download_file


if st.button('Get Result') and st.session_state.uploaded_file is not None:
    if st.session_state.model == "Cpx-Liq":
        md_frames = []
        for i in ['P20', 'J22', 'AL24']:
            if i == 'P20':
                input_data = st.session_state.data.loc[:, tr_dict['P20']['Cpx-Liq'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['P20']['Cpx-Liq'].values),
                                        columns=['MD_to_P20_Cpx_Liq'])
            elif i == 'J22':
                input_data = st.session_state.data.loc[:, tr_dict['J22']['Cpx-Liq'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['J22']['Cpx-Liq'].values),
                                        columns=['MD_to_J22_Cpx_Liq'])
            else:  # AL24
                input_data = st.session_state.data.loc[:, tr_dict['AL24']['Cpx-Liq'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['AL24']['Cpx-Liq'].values),
                                        columns=['MD_to_AL24_Cpx_Liq'])
            md_frames.append(md_value)

        md_result = pd.concat(md_frames, axis=1)

    elif st.session_state.model == 'Cpx-only':
        md_frames = []
        for i in ['P20', 'H21', 'J22', 'C23', 'AL24']:
            if i == 'P20':
                input_data = st.session_state.data.loc[:, tr_dict['P20']['Cpx-only'].columns]  # 修正：使用 P20 而不是 J22
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['P20']['Cpx-only'].values),
                                        columns=['MD_to_P20_Cpx_only'])
            elif i == 'H21':
                input_data = st.session_state.data.loc[:, tr_dict['H21']['Cpx-only'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['H21']['Cpx-only'].values),
                                        columns=['MD_to_H21_Cpx_only'])
            elif i == 'J22':
                input_data = st.session_state.data.loc[:, tr_dict['J22']['Cpx-only'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['J22']['Cpx-only'].values),
                                        columns=['MD_to_J22_Cpx_only'])
            elif i == 'C23':
                input_data = st.session_state.data.loc[:, tr_dict['C23']['Cpx-only'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['C23']['Cpx-only'].values),
                                        columns=['MD_to_C23_Cpx_only'])
            else:
                input_data = st.session_state.data.loc[:, tr_dict['AL24']['Cpx-only'].columns]
                md_value = pd.DataFrame(Cal_Madistance(input_data.values, tr_dict['AL24']['Cpx-only'].values),
                                        columns=['MD_to_AL24_Cpx_only'])
            md_frames.append(md_value)

        md_result = pd.concat(md_frames, axis=1)

    st.session_state.recomm = pd.concat(
        [st.session_state.data.iloc[:, 0], Cal_Recommendation_score(md_result, st.session_state.model)], axis=1)

    st.dataframe(st.session_state.recomm)

    download_file = to_result_df(st.session_state.recomm)
    st.download_button(
        label="Download",
        data=download_file,
        file_name=st.session_state.uploaded_file.name.split('.')[0] + "_Recommendation_Result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_results_button_1"
    )

elif st.session_state.uploaded_file is None:
    st.text(':red[Please upload your data firstly.]')
