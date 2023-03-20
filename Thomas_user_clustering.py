#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:59:53 2023

@author: hannah
"""


import os
import pandas as pd
from typing import List
import pickle
import re
import pandas as pd
from typing import List
import json
import psycopg2
import numpy as np
from torch.utils.data import DataLoader
import pickle
import torch
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd 
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import json
import numpy as np
from collections import Counter
import umap
from sklearn.preprocessing import StandardScaler
import hdbscan
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from functools import partial
from bertopic import BERTopic


def filtering_users(data, user_colname, user_list: List):
    '''
    Parameters
    ----------
    data : Pandas dataframe
        DESCRIPTION.
    user_colname : string
        DESCRIPTION.
    user_list : List
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    for user in user_list:
        data = data[data[user_colname] != user]
    return data




def preprocess(document: str) -> str:
    # remove URL pattern 
    # 안녕https://m.naver.com하세요 -> 안녕하세요
    # pattern = r'(http|https|ftp)://(?:[-\w.]|[\w/]|[\w\?]|[\w:])+'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove tag pattern
    # # 안녕<tag>하세요 -> 안녕하세요
    # pattern = r'<[^>]*>'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove () and inside of ()
    # # 안녕(parenthese)하세요 -> 안녕하세요
    # pattern = r'\([^)]*\)'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove [] and inside of []
    # # 안녕[parenthese]하세요 -> 안녕하세요
    # pattern = r'\[[^\]]*\]'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove special chars without comma and dot
    # # 안녕!!@@하세요, 저는 !@#호준 입니다. -> 안녕하세요, 저는 호준 입니다.
    # pattern = r'[^\w\s - .|,]'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove list characters
    # # 안녕12.1하세요 -> 안녕하세요
    # pattern = r'[0-9]*[0-9]\.[0-9]*'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove korean consonant and vowel
    # # 안녕ㅏ하ㅡㄱ세요 -> 안녕하세요
    # pattern = r'([ㄱ-ㅎㅏ-ㅣ]+)'
    # document = re.sub(pattern=pattern, repl='', string=document)

    # # remove chinese letter
    # # 안녕山하세요 -> 안녕하세요
    # pattern = r'[一-龥]*'
    # document = re.sub(pattern=pattern, repl='', string=document)
    
    ##https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=realuv&logNo=220699272999
    pattern =r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]' 
    # pattern =r'[^가-힣ㄱ-ㅎㅏ-ㅣ]' ##숫자와 영어를뺴고싶은경우
    document = re.sub(pattern=pattern, repl=' ', string=document)
    
    # 영어 소문자로 변환
    # document = document.lower()

    # remove empty space
    document = document.strip()

    # make empty spcae size as only one
    document = ' '.join(document.split())
    
    return document

##invalid 데이터 필터링아웃하기
def filter_data(dataframe, cols):
    '''
    Purpose:
        filtering non-character values
        filtering invalid characters
    '''
    dataframe.dropna(axis=0, how='any', inplace=True, subset = cols)
    # dataframe = dataframe.dropna(subset= cols)
    for col in cols:
        dataframe[col] = [preprocess(str(text)) for text in dataframe[col]]
        dataframe = dataframe[dataframe[col] != ' ']
        dataframe = dataframe[dataframe[col] != 'nan']        
        dataframe = dataframe[dataframe[col] != ''] 
        dataframe = dataframe[dataframe[col] != '\n'] 
    return dataframe

def check_data(**kwargs):
    # data = ti.xcom_pull(key='data', task_ids=['raw_data_preprocess'])
    df = kwargs.get('df', '/opt/airflow/dags/data/link_cat_pik.csv') 
    processed_data_path = kwargs.get('processed_data_path', '/opt/airflow/dags/data/processed_data.csv') 
    data = pd.read_csv(df)
    # data=data.dropna(subset=['link_title', 'pik_title'])
    data=data.dropna(subset=['link_title'])
    data=data.dropna(subset=['pik_title'])
    
    data.to_csv(processed_data_path, index=False)
    print('processed_data shape is: ', data.shape)
    



'''
raw_data_path:
artifical_user_list_path: Non-natural users should be removed from recommendation.
processed_data_saving_path: A path to save processed pandas dataframe that will be constantly used.
'''



#model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')
# path = '/home/hojun/python/rec_airflow_aws/dags/data'   
path = '/home/hannah/rec_airflow_aws/dags/data'


linkhub = pd.read_csv(f'{path}/linkhub_link.csv')
piks = pd.read_csv(f'{path}/piks_pik.csv')
catego = pd.read_csv(f'{path}/piks_category.csv')
user_language = pd.read_csv(f'{path}/users_user.csv')
user_friends = pd.read_csv(f'{path}/users_following.csv')
activities_scrap = pd.read_csv(f'{path}/activities_scrap.csv')

# linkhub = pd.read_csv('/home/hojun/temp/data_development/linkhub_link.csv')
# piks = pd.read_csv('/home/hojun/temp/data_development/piks_pik.csv')
# catego = pd.read_csv('/home/hojun/temp/data_development/piks_category.csv')
# user_language = pd.read_csv('/home/hojun/temp/data_development/users_user.csv')
# user_friends = pd.read_csv('/home/hojun/temp/data_development/users_following.csv')
# with open('/home/hojun/temp/data_development/artificial_user_list', 'rb') as f:
#     artificial_users = pickle.load(f)
    

##processing with friends list
user_friends.rename(columns = {'from_user_id': 'user_id', 'to_user_id':'followed_user_id'}, inplace=True)
user_friends = user_friends[(user_friends['is_deleted'] == 'f') | (user_friends['is_deleted'] == False)]  #'f' instead of False
user_friends.to_csv(f'{path}/users_following.csv', index=False)


with open(f'{path}/artificial_user_list', 'rb') as f:
    artificial_users = pickle.load(f)
    


linkhub.rename(columns = {'title':'link_title'}, inplace=True)
linkhub = linkhub[(linkhub['is_deleted'] == 'f') | (linkhub['is_deleted'] == False)]


piks.rename(columns = {'title':'pik_title'}, inplace=True)
piks = piks[(piks['is_deleted'] == 'f') | (piks['is_deleted'] == False)]


catego.rename(columns = {'title': 'cat_title'}, inplace = True)
catego = catego[(catego['is_deleted'] == 'f') | (catego['is_deleted'] == False)]


##category 테이블에 user id가 있기 때문에 그 아이디와 유저의 언어설정환경을 조인한다.
catego = pd.merge(catego, user_language, how = 'inner', left_on ='user_id', right_on='id', suffixes=('', '_user'))


##pik_info와 cat_info 병합
piks_cats = pd.merge(catego, piks, how = 'inner', left_on = 'pik_id', right_on = 'id', suffixes=('_cat', '_pik'))
piks_cats.columns

# filtered_pik_cat  = piks_cats[(piks_cats['status'] == 'public') & (piks_cats['is_draft'] == 'f')] #'f' instead of False
public_private_filtered_pik_cat =  piks_cats[(piks_cats['is_draft'] == False) | (piks_cats['is_draft'] == 'f')]





##이렇게해서 public이고 draft가 아닌 픽만남게된다
# link_cat_pik = pd.merge(linkhub, filtered_pik_cat, how='inner', left_on='category_id', right_on='id_cat', suffixes=('_link', '_catpik'))
# link_cat_pik.columns
# link_cat_pik.dropna(subset=['link_title', 'pik_title'], how='any', inplace=True)
# link_cat_pik.rename(columns={'id':'link_id', 'created':'link_create_time'}, inplace=True)
# link_cat_pik.sort_values(['user_id', 'category_id', 'link_id'], ascending = True, inplace=True)


##공개 비공개 픽 모두지만 na값은 없앴다 (즉 링크가 없는 빈 픽이나 유저는 없앴다)
link_cat_pik = pd.merge(linkhub, public_private_filtered_pik_cat, how='inner', left_on='category_id', right_on='id_cat', suffixes=('_link', '_catpik'))
link_cat_pik.rename(columns={'id':'link_id', 'created':'link_create_time'}, inplace=True)
link_cat_pik.sort_values(['user_id', 'category_id', 'link_id'], ascending = True, inplace=True)  













slug1 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3085].iloc[0]
slug2 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3186].iloc[0]
slug3 = link_cat_pik['slug'][link_cat_pik['pik_id'] == 3188].iloc[0]

##특정 string이 포함된 행을 df에서찾기
dummy1 = link_cat_pik[link_cat_pik['slug'].str.contains(slug1)]
dummy2 = link_cat_pik[link_cat_pik['slug'].str.contains(slug2)]
dummy3 = link_cat_pik[link_cat_pik['slug'].str.contains(slug3)]

## 추출한 행을 df에서제거하기
link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy1.index)]
link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy2.index)]
link_cat_pik = link_cat_pik[~link_cat_pik.index.isin(dummy3.index)]

##제거된것을확인
link_cat_pik[link_cat_pik['slug'] == slug1]
link_cat_pik[link_cat_pik['slug'] == slug2]
link_cat_pik[link_cat_pik['slug'] == slug3]

# link_cat_pik.drop(['description', 'memo', 'url', 'is_draft_link', 'link_create_time', 'id_cat', 'created_cat', 'id_user', 'id_pik', 'slug', 'language', 'is_draft_catpik', 'created_pik'], axis=1, inplace=True)


link_cat_pik = link_cat_pik.astype({'pik_id' :'int', 'link_id':'int', 'category_id':'int'})    


link_cat_pik = filtering_users(link_cat_pik, 'user_id', artificial_users)      
link_cat_pik = filter_data(link_cat_pik,  ['link_title'])
link_cat_pik = filter_data(link_cat_pik,  ['pik_title'])


link_cat_pik[link_cat_pik['user_id'] == 242]




candidate_users = [
242,
349,
527,
494,
576,
741,
815,
846,
893,
979,
1013,
1059,
1089,
1118,
1154,
1470,
1478,
1534,
1612,
2641,
2672,
2685,
2981,
3017,
3057,
3066,
3070,
3166,
3192,
3413,
3436,
3569,
3571,
3572,
3651,
3760,
3832,
3858,
3928,
3945,
3973,
3999,
4020,
4028,
4039,
4053,
4083,
4092,
4129,
4146,
4190,
4208,
4210,
4212,
4224,
4252,
4264,
4288,
4338,
4372,
4406,
4407,
4425,
4465,
4473,
4476,
4482,
4523,
4535,
4570,
4628,
4640,
4646,
4679,
4717,
4734,
4773,
4813,
4844,
4873,
4912,
4915,
4925,
4975,
5040,
5077,
5079,
5147,
5190,
5607,
5688,
5839,
5896,
5911,
5997,
6028,
6060,
6070,
6102,
6105,
6108,
6184,
6194,
6216,
6350,
6476,
6515,
6524,
6536,
6538,
6548,
6550,
6554,
6586,
6624,
6636,
6715,
6732,
6754,
6900,
6908,
6909,
6944,
6961,
6979,
6991,
7013,
7043,
7105,
7140,
7147,
7159,
7237,
7240,
7254,
7255,
7299,
7315,
7319,
7329,
7330,
7340,
7344,
7375,
7386,
7443,
7451,
7479,
7483,
7490,
7492,
7503,
7518,
7528,
7539,
7547,
7569,
7578,
7598,
7633,
7681,
7693,
7694,
7706,
7710,
7718,
7748,
7749,
7798,
7820,
7822,
7830,
7863,
7880,
7891,
7953,
7955,
7992,
7993,
8034,
8137,
8195,
8204,
8275,
8464,
8494,
8501,
8543,
8550,
8552,
8613,
8651,
8662,
8683,
8706,
8740,
8748,
8752,
8764,
8780,
8788,
8789,
8806,
8808,
8848,
8858,
8867,
8893,
8900,
8909,
8921,
8927,
8958,
8967,
8969,
8975,
8978,
8980,
9006,
9013,
9022,
9029,
9111,
9332,
9358,
9399,
9402,
9561,
9576,
9586,
9587,
9612,
9621,
9622,
9629,
9632,
9690,
9750,
9764,
9832,
9840,
9843,
9860,
9861,
9862,
9871,
9874,
9880,
9958,
9977,
9980,
10112,
10125,
10133,
10139,
10140,
10204,
10211,
10251,
10258,
10271,
10330,
10339,
10370,
10384,
10390,
10413,
10415,
10438,
10464,
10468,
10472,
10480
]

##필요한 유저 값만 필터링해서 가져오기
data = link_cat_pik.loc[link_cat_pik['user_id'].isin(candidate_users)]
data.columns

##'memo'는 link_memo를 뜻함
##'status'는 공개/비공개
##'is_default_pik' true는 퀵링크박스인것
processed_data = data.loc[:,['user_id', 'pik_id', 'category_id', 'link_id', 'status', 'is_default_pik', 'pik_title','cat_title','link_title', 'memo']]  ##'memo'는 link_memo를 뜻함


##User-Link

def group_and_count(dataframe, groupby, groupwhat):
    
    groupby_result = dataframe[[groupby, groupwhat]].groupby(groupby)[groupwhat].apply(lambda x: x.tolist()).to_dict() #한 유저가 몇개의 voca(빌딩과카드번호로를 틀렸는지 다 보여준다 #groupby는 유저로그룹묶어주고 voca에대한값을보여달라는것
    group_counting = dataframe.groupby(groupby)[groupwhat].nunique().sort_values(ascending=False)
    return groupby_result, group_counting 

groupby_link, num_link_by_user = group_and_count(processed_data, 'user_id', 'link_id')
num_link_by_user.name = 'num_link'


##공개픽
public_piks, num_public_pik_by_user = group_and_count(processed_data[processed_data['status'] == 'public'], 'user_id', 'pik_id')
num_public_pik_by_user.name = 'num_public_pik'


##퀵링크박스는아니면서 비공개인 픽
private_noquiklink_piks, num_private_noquiklink_pik_by_user = group_and_count(processed_data[(processed_data['status'] == 'private') & (processed_data['is_default_pik'] == False)], 'user_id', 'pik_id')
num_private_noquiklink_pik_by_user.name = 'num_private_noquinklink_pik'


##전체 카테고리 수
groupby_category, num_cat_by_user = group_and_count(processed_data, 'user_id', 'category_id')
num_cat_by_user.name = 'num_cat'


##메모가 등록되지 않은 링크 수 
NA_indices = pd.isna(processed_data['memo'])[pd.isna(processed_data['memo']) == True].index
groupby_nomemo_link, num_nomemo_link_by_user = group_and_count(processed_data.filter(items = NA_indices, axis = 0), 'user_id', 'link_id')
num_nomemo_link_by_user.name = 'num_nomemo_link'


##메모가 등록된 링크 수 
none_NA_indices = pd.isna(processed_data['memo'])[pd.isna(processed_data['memo']) == False].index
groupby_with_memo_link, num_with_memo_link_by_user = group_and_count(processed_data.filter(items = none_NA_indices, axis = 0), 'user_id', 'link_id')
num_with_memo_link_by_user.name = 'num_with_memo_link'

##퀵링크박스에 등록된 링크 수 
groupby_quicklink, num_quicklink_by_user = group_and_count(processed_data[processed_data['is_default_pik'] == True], 'user_id', 'link_id')
num_quicklink_by_user.name = 'num_quicklink'


##각유저의 링크 스크랩 수 
activities_scrap = activities_scrap[activities_scrap['content_type_id'] == 24]
groupby_scrap_link, num_scrap_by_user = group_and_count(activities_scrap, 'user_id', 'object_id')
num_scrap_by_user.name = 'num_scraplink'



processed_data = processed_data.merge(num_link_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_public_pik_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_private_noquiklink_pik_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_cat_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_nomemo_link_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_with_memo_link_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_quicklink_by_user, how='left', on ='user_id')
processed_data = processed_data.merge(num_scrap_by_user, how='left', on ='user_id')


processed_data.columns

for col in processed_data.columns[10:]:
    processed_data[col] = processed_data[col].replace(np.nan, 0)


processed_data.columns
unique_processed_data = processed_data.drop_duplicates(subset=['user_id','num_link','num_public_pik','num_private_noquinklink_pik','num_cat','num_nomemo_link', 'num_with_memo_link', 'num_quicklink', 'num_scraplink'])

unique_processed_data.columns
unique_processed_data_ = unique_processed_data.drop(columns = ['user_id','pik_id','category_id','link_id', 'status', 'is_default_pik', 'pik_title', 'cat_title','link_title','memo'])

scaled_unique_processed_data = StandardScaler().fit_transform(unique_processed_data_)


##Torch data화 시키기



def process_sentence(data, tokenizer, tokenizing_col: str, max_len:int=24, return_tensors='pt', padding='max_length', truncation=True):
    # initialize dictionary to store tokenized sentences for link title
    tokens = {'input_ids': [], 'attention_mask': []}
    
    for sentence in tqdm(data[tokenizing_col]):
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=24,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    return tokens





class PQDataset_torch(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        
    def __getitem__(self, index): 
        return self.input_ids[index], self.attention_masks[index]
        
    def __len__(self): 
        return self.input_ids.shape[0]



def load_model(model_name):
    model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)    
    return tokenizer



#model_name = kwargs.get('model_name', 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking')
tokenizer_name = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
# processed_data_path = '/home/hannah/python/Pikurate_services/Thomas_user_clustering/data/processed_data.csv'
tokenizing_col =  'link_title' ##or pik_title
max_len =  24
return_tensors =  'pt'
padding =  'max_length'
truncation =  True
batch_size =  256
saving_dataloader_path = f'/home/hannah/python/Pikurate_services/Thomas_user_clustering/data/{tokenizing_col}_dataloader.pickle'

tokenizer = load_tokenizer(tokenizer_name)
# processed_data = pd.read_csv(processed_data_path)
# processed_data=processed_data.dropna(subset=['link_title', 'pik_title'])
print(processed_data.shape)


tokens = process_sentence(processed_data, tokenizer, tokenizing_col, max_len, return_tensors, padding, truncation)
dataset = PQDataset_torch(tokens['input_ids'], tokens['attention_mask'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
with open(saving_dataloader_path,'wb') as f:
    pickle.dump(dataloader, f)



tokenizing_col =  'pik_title' ##or pik_title
saving_dataloader_path = f'/home/hannah/python/Pikurate_services/Thomas_user_clustering/data/{tokenizing_col}_dataloader.pickle'
tokens = process_sentence(processed_data, tokenizer, tokenizing_col, max_len, return_tensors, padding, truncation)
dataset = PQDataset_torch(tokens['input_ids'], tokens['attention_mask'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
with open(saving_dataloader_path,'wb') as f:
    pickle.dump(dataloader, f)
    
    




##Text embedding화 시키기 




def load_tokenizer_and_model(tokenizer_name, model_name):
    
    '''
    embedding을사용하려면 nli모델을사용해야한다
    '''
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model
    


##링크 타이틀과 픽 타이틀 엠베딩이 만들어졌지만 픽을 구성하는 엠베딩을 아직 계산하지는 않았기 때문에 링크를 픽별로 그룹화 해줌 
##유저/픽별로 모든 링크를수집해서 딕셔너리화하기        
def get_links_by(dataframe, groupby, groupwhat):
    
    groupby_link = dataframe[[groupby, groupwhat]].groupby(groupby)[groupwhat].apply(lambda x: x.tolist()).to_dict() #한 유저가 몇개의 voca(빌딩과카드번호로를 틀렸는지 다 보여준다 #groupby는 유저로그룹묶어주고 voca에대한값을보여달라는것
    return groupby_link  





#################### 두개이상틀린것에대해서 유사도측정####################################

def get_vectors(first_map, second_map):
    first_vec  = dict()
    for uid, links in first_map.items(): #voca와 voca안에 있는 text로 나누어줌 
        temp = list()
        for element in links:
            try:
                temp.append(second_map[element]) #voca에 있는 text의 각 단어들을 모두 append 하는 것 
            except KeyError:
                pass
        first_vec[uid] = np.mean(temp, axis=0)  #append된 모든 단어들의 값을 mean으로 평균내어 주는 것 
    
    return first_vec




##여기서는 한번은 link관련해서 코드실행해주고 한번은 pik관련해서 코드실행해줘야한다

default_path = '/home/hannah/python/Pikurate_services/Thomas_user_clustering'
# processed_data_path = kwargs.get('processed_data_path', '/opt/airflow/dags/data/processed_data.csv')   
tokenizer_name =  'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
model_name =  'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
dataloader_path =  '/home/hannah/python/Pikurate_services/Thomas_user_clustering/data/link_title_dataloader.pickle'
# dataloader_path =  '/home/hannah/python/Pikurate_services/Thomas_user_clustering/data/pik_title_dataloader.pickle'
which_emb =  'linktitle_emb'
# which_emb =  'piktitle_emb'
link_rec_on =  False
device =  'cuda' 

# processed_data = pd.read_csv(processed_data_path)
tokenizer, model = load_tokenizer_and_model(tokenizer_name, model_name)
with open(dataloader_path, 'rb') as f:
    dataloader = pickle.load(f)

model.eval()
mean_pooled_total = []
if model_name == 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking':
        
    
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_att_masks = batch
            
        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
     
        with torch.no_grad():
                outputs = model(b_input_ids, attention_mask = b_att_masks) ##For distilbert we dont need token_type_ids
                # outputs.keys()
                embeddings = outputs.last_hidden_state
                mask = b_att_masks.unsqueeze(-1).expand(embeddings.size()).float()
                masked_embeddings = embeddings * mask
                summed_emb = torch.sum(masked_embeddings, 1)
                summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled_emb = summed_emb / summed_mask
                # mean_pooled_emb = F.normalize(mean_pooled_emb, p=2, dim=1)

        mean_pooled_total.append(mean_pooled_emb)
        
else:
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_att_masks = batch
            
        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers. 
     
        with torch.no_grad():
                outputs = model(b_input_ids, attention_mask = b_att_masks, token_type_ids=None)
                outputs.keys()
                # pooled_outputs = outputs[1]  ##(last_hidden_state, pooler_output, hidden_states[optional], attentions[optional])
                embeddings = outputs.last_hidden_state
                mask = b_att_masks.unsqueeze(-1).expand(embeddings.size()).float()
                masked_embeddings = embeddings * mask
                summed_emb = torch.sum(masked_embeddings, 1)
                summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled_emb = summed_emb / summed_mask
        
        mean_pooled_total.append(mean_pooled_emb)
        
if which_emb == 'linktitle_emb':
    link_final_pred = torch.cat(mean_pooled_total, 0).detach().cpu().numpy()
    link_vectors = dict(zip(processed_data.link_id, link_final_pred))  ##link title vectors

    if link_rec_on:
        '''link_vec을 저장하기 위해서는 아래 코드들을 언코멘트 해준다'''
        # 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
        link_vectors_tolist = {str(k): v.tolist() for k, v in link_vectors.items()}
        with open(f"{default_path}/data/{which_emb}_vec.json", "w") as f: ##2G가까이되는 큰 데이터이기 때문에 왠만하면 세이브하지말자
            json.dump(link_vectors_tolist, f)

   

    ## keys: pik, values: link_id ##pik_id로 link를 그룹화해라라는뜻
    ##pik추천을 위한 것
    pik_link = get_links_by(processed_data, 'pik_id', 'link_id')
    with open(f'{default_path}/data/pik_link.json', 'w') as f:
        json.dump(pik_link, f)
        
    pik_vec = get_vectors(pik_link, link_vectors) #유저가 틀린 문장들이계산되었는데 그문장들을 모두모아서에버리지하는것. 유저가틀린모든문장을모아서평균계산
    
    ## 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
    pik_vec_tolist = {str(k):v.tolist() for k,v in pik_vec.items()}
    with open(f"{default_path}/data/pik_vec.json", "w") as f:
        json.dump(pik_vec_tolist, f)
    
    ## 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
    ##pik을 기준으로 link를 딕셔너리로 정렬시키고 그것을 json으로 저장하라 
    num_link_by_pik = processed_data.groupby('pik_id')['link_id'].nunique().sort_values(ascending=False)
    num_link_by_pik = {str(key):value for key,value in num_link_by_pik.items()} ##딕셔너리화한다
    with open(f'{default_path}/data/num_link_by_pik.json', 'w') as f:
        json.dump(num_link_by_pik, f)


    user_pik = get_links_by(processed_data, 'user_id', 'pik_id')
    with open(f'{default_path}/data/user_pik.json', 'w') as f:
        json.dump(user_pik, f)
    

    ##유저추천을 위한 것
    num_link_by_user = processed_data.groupby('user_id')['link_id'].nunique().sort_values(ascending=False)
    num_link_by_user = {str(key):value for key,value in num_link_by_user.items()} ##딕셔너리화한다 
    with open(f'{default_path}/data/num_link_by_user.json', 'w') as f:
        json.dump(num_link_by_user, f)

    user_link = get_links_by(processed_data, 'user_id', 'link_id')
    ##for user-rec
    user_vec = get_vectors(user_link, link_vectors)
    user_vec_tolist = {str(k):v.tolist() for k,v in user_vec.items()}
    with open(f"{default_path}/data/user_vec.json", "w") as f:
        json.dump(user_vec_tolist, f)


    user_sent = get_links_by(processed_data, 'user_id', 'link_title')

    
    with open(f'{default_path}/data/user_link.json', 'w') as f:
        json.dump(user_link, f)



    
    # with open(f'{default_path}/data/linkid_title_dict.json') as f:
    #     linkid_title_dict = json.load(f)
    
    # with open(f'{default_path}/data/pikid_title_dict.json') as f:
    #     pikid_title_dict = json.load(f)



    # user_lang_dict_detected = {}
    # for user_id in user_link.keys():
    #     ##predict user language
    #     lang_pred_user = [fmodel.predict([linkid_title_dict[str(link_id)]])[0][0][0][-2:] for link_id in user_link[user_id]]
    #     language_pred_count_user_dict = Counter(lang_pred_user)
    #     final_pred_lang_user = [k for k, v in language_pred_count_user_dict.items() if v == max(language_pred_count_user_dict.values())][0]
    #     user_lang_dict_detected[user_id] = final_pred_lang_user

    # with open(f'{default_path}/data/user_lang_dict_detected.json', 'w') as f:
    #     json.dump(user_lang_dict_detected, f)



    # pik_lang_dict_detected = {}    
    # for pik_id in pik_link.keys():
    #     ##predict pik language
    #     lang_pred_pik = [fmodel.predict([linkid_title_dict[str(link_id)]])[0][0][0][-2:] for link_id in pik_link[pik_id]]
    #     language_pred_count_pik_dict = Counter(lang_pred_pik)
    #     final_pred_lang_pik = [k for k, v in language_pred_count_pik_dict.items() if v == max(language_pred_count_pik_dict.values())][0]
    #     pik_lang_dict_detected[pik_id] = final_pred_lang_pik
    
    # with open(f'{default_path}/data/pik_lang_dict_detected.json', 'w') as f:
    #     json.dump(pik_lang_dict_detected, f)    



    
elif which_emb == 'piktitle_emb':
    pik_final_pred = torch.cat(mean_pooled_total, 0).detach().cpu().numpy()
    piktitle_vectors = dict(zip(processed_data.pik_id, pik_final_pred))  ##pik title vectors

    ## 아래 코드는 BentoML에서 실행하기 위해 중요하다. Bento는 json데이터에 가장 친숙하기 때문에 왠만해선 json을 쓰도록하자
    piktitle_vectors_tolist = {str(k): v.tolist() for k, v in piktitle_vectors.items()}
    with open(f"{default_path}/data/{which_emb}_vec.json", "w") as f: ##2G가까이되는 큰 데이터이기 때문에 왠만하면 세이브하지말자
        json.dump(piktitle_vectors_tolist, f)

scaled_unique_processed_data
user_vecs =np.stack(user_vec.values(), axis =0)


##Transform user_vec to reduce dimensions by UMAP
# umap_embeddings = (umap.UMAP(n_neighbors=15, 
#                             n_components=7, 
#                             metric='cosine', 
#                             random_state=2023)
#                         .fit_transform(user_vecs))

data_before_umap = np.concatenate([scaled_unique_processed_data, user_vecs], axis=1)


# scaled_unique_processed_data.shape
# umap_embeddings.shape
# zz = np.concatenate([scaled_unique_processed_data, umap_embeddings], axis=1)

# scaled_unique_processed_data[0]
# umap_embeddings[0]

# final_data = np.concatenate([scaled_unique_processed_data, umap_embeddings], axis=1)


# clusterer = hdbscan.HDBSCAN(min_cluster_size = 5,
#                            metric='euclidean', 
#                            cluster_selection_method='eom').fit(final_data)

# clusterer.labels_
# dir(clusterer)


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state)
                            .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters




def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns the label count and cost of a given clustering

    Arguments:
        clusters: HDBSCAN clustering object
        prob_threshold: float, probability threshold to use for deciding
                        what cluster labels are considered low confidence

    Returns:
        label_count: int, number of unique cluster labels, including noise
        cost: float, fraction of data points whose cluster assignment has
              a probability below cutoff threshold
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost


# labels_def, cost_def = score_clusters(clusterer)




# color_palette = sns.color_palette('deep', 8)
# cluster_colors = [color_palette[x] if x >= 0
#                   else (0.5, 0.5, 0.5)
#                   for x in clusters.labels_]
# cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                          zip(cluster_colors, clusters.probabilities_)]
# s_len = len(clusters.labels_)
# final_data.T
# plt.scatter(*final_data.T, s=s_len, linewidth=0, c=cluster_member_colors, alpha=0.25)
# plt.show


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize

    Arguments:
        params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'random_state' and
               their values to use for evaluation
        embeddings: embeddings to use
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters

    Returns:
        loss: cost function result incorporating penalties for falling
              outside desired range for number of clusters
        label_count: int, number of unique cluster labels, including noise
        status: string, hypoeropt status

        """
    
    clusters = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 random_state = params['random_state'])
    
    label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
    
    #15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15 
    else:
        penalty = 0
    
    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


# In[16]:


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

        """
    
    trials = Trials()
    fmin_objective = partial(objective, 
                             embeddings=embeddings, 
                             label_lower=label_lower,
                             label_upper=label_upper)
    
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=max_evals, 
                trials=trials)

    best_params = space_eval(space, best)
    print ('best:')
    print (best_params)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters = generate_clusters(embeddings, 
                                      n_neighbors = best_params['n_neighbors'], 
                                      n_components = best_params['n_components'], 
                                      min_cluster_size = best_params['min_cluster_size'],
                                      random_state = best_params['random_state'])
    
    return best_params, best_clusters, trials




hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(6,15)),
    "n_components": hp.choice('n_components', range(6,20)),
    "min_cluster_size": hp.choice('min_cluster_size', range(7,13)),
    "random_state": 777
}

label_lower = 3
label_upper = 15
max_evals = 400




best_params_use, best_clusters_use, trials_use = bayesian_search(data_before_umap, 
                                                                 space=hspace, 
                                                                 label_lower=label_lower, 
                                                                 label_upper=label_upper, 
                                                                 max_evals=max_evals)





umap_embeddings = (umap.UMAP(n_neighbors=6, 
                            n_components=16, 
                            metric='cosine', 
                            random_state=777)
                        .fit_transform(data_before_umap))




clusterer = hdbscan.HDBSCAN(min_cluster_size = 7,
                           metric='euclidean', 
                           cluster_selection_method='eom').fit(umap_embeddings)



clusterer.labels_
dir(clusterer)











### 비쥬얼라이제이션 ###

import seaborn as sns
import matplotlib.pyplot as plt
# palette = sns.color_palette()
# cluster_colors = [sns.desaturate(palette[col], sat)
#                   if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
#                   zip(clusters.labels_, clusters.probabilities_)]

# plot_kwds = {'alpha' : 0.7, 's' : 80, 'linewidths':0}
# plt.scatter(final_data.T[], final_data.T[1], c=cluster_colors, **plot_kwds)


# plt.scatter(*final_data.T, s=50, linewidth=0, c='b', alpha=0.25)


plot_kwds = {'alpha' : 0.25, 's' : 10, 'linewidths':0}


from sklearn.manifold import TSNE

projection = TSNE().fit_transform(umap_embeddings)
plt.scatter(*projection.T, **plot_kwds)


color_palette = sns.color_palette('Paired', 12)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]

plt.scatter(*projection.T, s=30, linewidth=0, c=cluster_member_colors, alpha=0.5)






















data_present = unique_processed_data.drop(columns = ['pik_id','category_id','link_id', 'status', 'is_default_pik', 'pik_title', 'cat_title','link_title','memo'])



# https://stackoverflow.com/questions/901412/python-joining-multiple-lists-to-one-single-sentence
# result = ' '.join(sum(user_sent.values(), []))


total_sentences = ' '.join(sum(user_sent.values(), []))

##각 유저마다 가지고 있는 링크 타이틀의 단어들을 모두 같다 붙인것 
user_sents_dict = {}
for user_id, user_sents in user_sent.items():
    user_sents_dict[user_id] = ' '.join(user_sents)
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import *

def tf_idf(data, max_df=0.85, smooth_idf=True, use_idf=True):
    ##making TF
    ##우리가 가지고 있는 전체문장으로 tf-idf를 트레이닝한다.
    if isinstance(data, pd.Series):
        print('It is pandas.Series type')
        data = data.tolist()
    else:
        pass
    cv=CountVectorizer(max_df=max_df)    
    word_count_vector = cv.fit_transform(data)
    # list(cv.vocabulary_.keys())[:10]
    # feature_names=cv.get_feature_names()
    feature_names=cv.get_feature_names_out()
    
    ##making IDF
    tfidf_transformer = TfidfTransformer(smooth_idf=smooth_idf, use_idf=use_idf)
    tfidf_transformer.fit(word_count_vector)    
        
    
    return cv, tfidf_transformer, feature_names

count_vectorizer, tfidf_transformer, feature_names = tf_idf(link_cat_pik['link_title'])









def tf_idf_to_user_keywords(count_vectorizer, tfidf_transformer, user_words):

    
    tf_idf_vector = tfidf_transformer.transform(count_vectorizer.transform([user_words]))
    
    return tf_idf_vector


user_tf_idf_vector_dict = {}
for userid, allwordsbyuser in user_sents_dict.items():
    user_tf_idf_vector_dict[userid] = tf_idf_to_user_keywords(count_vectorizer, tfidf_transformer, allwordsbyuser)

    
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


# sorted_keywords=sort_coo(tf_idf_vector.tocoo())



def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results



user_topkeywords = {}
for userid, tf_idf_vector in user_tf_idf_vector_dict.items(): 
    sorted_keywords=sort_coo(tf_idf_vector.tocoo())
    keywords=extract_topn_from_vector(feature_names,sorted_keywords, 10)
    user_topkeywords[userid] = keywords


user_topkeywords_series = pd.Series(user_topkeywords)
user_topkeywords_series.name = 'topkeywords'



len(clusterer.labels_)
np.unique(clusterer.labels_)
data_present = data_present.merge(user_topkeywords_series, left_on='user_id', right_index=True)
data_present['cluster'] = clusterer.labels_


data_present.to_csv('/home/hannah/python/Pikurate_services/Thomas_user_clustering/data/thomas_clustering_data.csv', index=False)
# user_linkstext = {}
# for user, links in user_link.items():
#     for link in links:
        
#         # user_linkstext[user] = link_cat_pik['link_title'][link_cat_pik['link_id'] == link]
        

# z = link_cat_pik[link_cat_pik['user_id'] == 10480]
# z.columns


