import json
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer

from lib.make_dataset import read_pure_data


def row_to_sentences(row, raw_config, with_label: bool):
    real_data_path = raw_config['real_data_path']
    dataset_name = real_data_path.split('/')[-2]
    with open(f'{real_data_path}\info.json', 'r') as file:
        config = json.load(file)

    if dataset_name == 'adult':
        sentences = [sen.format(workclass=row['workclass'],
                                education=row['education'],
                                marital_status=row['marital-status'],
                                occupation=row['occupation'],
                                relationship=row['relationship'],
                                race=row['race'],
                                sex=row['sex'],
                                native_country=row['native-country']) for sen in config['cat_prompt']]
        if with_label:
            salary = 'greater than or equal to 50k' if row['salary'] == '1' else 'less than 50k'
            label_prompt = [config['label_prompt'].format(salary=salary)]
            sentences = label_prompt + sentences

    elif dataset_name == 'shopper':
        sentences = [sen.format(Month=row['Month'],
                                OperatingSystems=row['OperatingSystems'],
                                Browser=row['Browser'],
                                Region=row['Region'],
                                TrafficType=row['TrafficType'],
                                VisitorType=row['VisitorType'],
                                Weekend='is' if str(row['Weekend'])=='True' else 'is not') for sen in config['cat_prompt']]
        if with_label:
            revenue = 'completed' if str(row['Revenue'])=='True' else 'did not complete'
            label_prompt = [config['label_prompt'].format(Revenue=revenue)]
            sentences = label_prompt + sentences

    elif dataset_name == 'buddy':
        sentences = [sen.format(condition=row['condition'],
                                color_type=row['color_type'],
                                X1=row['X1'],
                                X2=row['X2'],
                                breed_category=row['breed_category']) for sen in config['cat_prompt']]
        if with_label:
            label_prompt = [config['label_prompt'].format(pet_category=row['pet_category'])]
            sentences = label_prompt + sentences

    elif dataset_name == 'obesity':
        sentences = [sen.format(gender=row['Gender'],
                                his_over=row['family_history_with_overweight'],
                                FAVC=row['FAVC'],
                                CAEC=row['CAEC'],
                                SMOKE=row['SMOKE'],
                                SCC=row['SCC'],
                                CALC=row['CALC'],
                                MTRANS=row['MTRANS']) for sen in config['cat_prompt']]
        if with_label:
            label_prompt = [config['label_prompt'].format(NObeyesdad=row['NObeyesdad'])]
            sentences = label_prompt + sentences

    elif dataset_name == 'magic':
        sentences = []
        if with_label:
            Class = config['Class'][str(row['class'])]
            label_prompt = [config['label_prompt'].format(Class=Class)]
            sentences = label_prompt + sentences

    elif dataset_name == 'bean':
        sentences = []
        if with_label:
            label_prompt = [config['label_prompt'].format(Class=row['Class'])]
            sentences = label_prompt + sentences

    elif dataset_name == 'page':
        sentences = []
        if with_label:
            Class = config['Class'][str(row['Class'])]
            label_prompt = [config['label_prompt'].format(Class=Class)]
            sentences = label_prompt + sentences

    elif dataset_name == 'churn':
        HasCrCard = "holds" if str(row['HasCrCard']) == "1" else "doesn't hold"
        IsActiveMember = "is" if str(row['IsActiveMember']) == "1" else "is not"
        sentences = [sen.format(Geography=row['Geography'],
                                Gender=row['Gender'],
                                NumOfProducts=row['NumOfProducts'],
                                HasCrCard=HasCrCard,
                                IsActiveMember=IsActiveMember) for sen in config['cat_prompt']]
        if with_label:
            label_prompt = ["the customer closed account" if str(row['Exited']) == "1" else "the customer is retained"]
            sentences = label_prompt + sentences

    elif dataset_name == 'abalone':
        sentences = []
        if with_label:
            Sex = config['Sex'][str(row['Sex'])]
            label_prompt = [config['label_prompt'].format(Sex=Sex)]
            sentences = label_prompt + sentences

    elif dataset_name == 'bike':
        sentences = [sen.format(season=row['season'],
                                yr=row['yr'],
                                mnth=row['mnth'],
                                hr=row['hr'],
                                holiday='is' if str(row['holiday']) == "1" else 'is not',
                                workingday='is' if str(row['workingday']) == "1" else 'is not',
                                weathersit=row['weathersit'],) for sen in config['cat_prompt']]
        if with_label:
            label_prompt = [config['label_prompt'].format(weekday=row['weekday'])]
            sentences = label_prompt + sentences

    elif dataset_name == 'insurance':
        sentences = [sen.format(sex=row['sex'],
                                children=row['children'],
                                smoker=row['smoker'],) for sen in config['cat_prompt']]
        if with_label:
            label_prompt = [config['label_prompt'].format(region=row['region'])]
            sentences = label_prompt + sentences

    elif dataset_name == 'productivity':
        sentences = [sen.format(quarter=row['quarter'],
                                department=row['department'],) for sen in config['cat_prompt']]
        if with_label:
            label_prompt = [config['label_prompt'].format(day=row['day'])]
            sentences = label_prompt + sentences

    elif dataset_name == 'covertype':
        wilderness_areas = config['wilderness_areas'][str(int(row['Wilderness_Area']))]
        soil_types = config['soil_types'][str(int(row['Soil_Type']))]
        sentences = [sen.format(wilderness_areas=wilderness_areas,
                                soil_types=soil_types, ) for sen in config['cat_prompt']]
        if with_label:
            cover_type = config['cover_type'][str(int(row['Cover_Type']))]
            label_prompt = [config['label_prompt'].format(cover_type=cover_type)]
            sentences = label_prompt + sentences

    else:
        raise ValueError("wrong real data path!")


    return sentences


def make_dataset_and_encode(raw_config, berttokenizer, bertmodel, device, with_label, split='train'):
    with torch.no_grad():
        # 读取并准备数据
        X_num, X_cat, y = read_pure_data(raw_config['real_data_path'], raw_config, split)
        X_cat = X_cat if X_cat is not None else np.empty((X_num.shape[0], 0))
        if y.ndim == 1:
            y = y[:, None]
        X_num_columns = raw_config['X_num_columns']
        X_cat_columns = raw_config['X_cat_columns']
        y_column = raw_config['y_column']
        df = pd.DataFrame(np.concatenate((X_num, X_cat, y), axis=1), columns=X_num_columns + X_cat_columns + y_column)

        # all_last_hidden_states = []
        all_pooler_outputs = []

        # 对所有行进行处理，并将编码的结果存储
        for _, row in df.iterrows():
            sentences = row_to_sentences(row, raw_config, with_label)

            # Tokenize
            max_length = raw_config['model_params'].get('bert_max_length', 13)
            tokens = berttokenizer(sentences, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")

            input_ids = tokens['input_ids'].to(device)  # shape: (8, t)
            attention_mask = tokens['attention_mask'].to(device)  # shape: (8, t)

            # Pass through BERT model
            hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

            # all_last_hidden_states.append(last_hidden_state)
            all_pooler_outputs.append(cls_head.to(device))

        # Convert list of tensors to a single tensor
        # all_last_hidden_states = torch.stack(all_last_hidden_states)  # shape: (n, 15, t, hidden_size)
        all_pooler_outputs = torch.stack(all_pooler_outputs)  # shape: (n, 15, hidden_size)

    return all_pooler_outputs


def sum_cls_head_mask(cls_head, device, if_mask):
    '''
    对于adult数据集而言，一条数据对应15个句子。
    本方法用于求 15 个句子经过bert编码后的 15个 cls_head 的和。
    但是，会随机将 15 个句子进行 mask，是为了将15个特征进行排列组合。
    '''
    if if_mask:
        mask = torch.bernoulli(torch.full((cls_head.shape[0], cls_head.shape[1], 1), 0.5, device=device))
        mask[:, 0] = 1     # 第一列特征永远不mask，也就是lable
    else:
        mask = torch.ones_like(cls_head, device=device)
        mask[:, 0] = 1
    # print(mask)
    cls_head_mask = cls_head * mask
    sum_cls = torch.sum(cls_head_mask, dim=1)
    label_cls = cls_head[:, 0]   # 第一列特征

    return sum_cls, label_cls



def default_sentences(label, dataset, raw_config):
    if raw_config['model_params']['bert'] == 'bert-base-uncased':
        berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 768
        bertmodel = BertModel.from_pretrained('bert-base-uncased')

    elif raw_config['model_params']['bert'] == 'huawei-noah/TinyBERT_General_4L_312D':
        berttokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')  # 312
        bertmodel = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

    elif raw_config['model_params']['bert'] == 'prajjwal1/bert-tiny':
        # berttokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')  # 128
        # bertmodel = BertModel.from_pretrained('prajjwal1/bert-tiny')

        berttokenizer = BertTokenizer.from_pretrained('D:\Study\自学\表格数据生成\models\prajjwal1-bert-tiny')
        bertmodel = BertModel.from_pretrained('D:\Study\自学\表格数据生成\models\prajjwal1-bert-tiny')

    else:
        raise ValueError("wrong bert name!")


    if raw_config['Transform']['y_plus1']:
        label = label - 1

    real_data_path = raw_config['real_data_path']
    dataset_name = real_data_path.split('/')[-2]
    with torch.no_grad():
        sen_list = []
        if dataset_name == "adult":
            # sen1 = ['relationship is Husband', 'sex is Male', 'salary is less than 50k']
            # sen2 = ['relationship is Husband', 'sex is Male', 'salary is greater than or equal to 50k']
            # sen0 = ['salary is less than 50k', 'relationship is Wife', 'sex is Female']
            # sen1 = ['salary is greater than or equal to 50k', 'relationship is Wife', 'sex is Female']
            # sen0 = ['salary is less than 50k', 'sex is Female']
            # sen1 = ['salary is greater than or equal to 50k', 'sex is Female']
            # sen0 = ['salary is less than 50k', 'Workclass is State-gov']
            # sen1 = ['salary is greater than or equal to 50k', 'Workclass is State-gov']
            sen0 = ['salary is less than 50k']
            sen1 = ['salary is greater than or equal to 50k']
            # sen0 = ['salary is less than 50k', 'relationship is Wife', 'sex is Female', 'Workclass is Private']
            # sen1 = ['salary is greater than or equal to 50k', 'relationship is Wife', 'sex is Female', 'Workclass is Private']
            # sen0 = ['salary is less than 50k',
            #         'relationship is Husband',
            #         'sex is Male',
            #         'Workclass is Private',
            #         'Education is Masters',
            #         'marital-status is Married-civ-spouse',
            #         'Occupation is Exec-managerial',
            #         'race is White',
            #         'native-country is United-States']
            # sen1 = ['salary is greater than or equal to 50k',
            #         'relationship is Husband',
            #         'sex is Male',
            #         'Workclass is Private',
            #         'Education is Masters',
            #         'marital-status is Married-civ-spouse',
            #         'Occupation is Exec-managerial',
            #         'race is White',
            #         'native-country is United-States']

            sen_list.append(sen0)
            sen_list.append(sen1)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'0': cls_heads[0], '1': cls_heads[1]}

        elif dataset_name == "shopper":
            sen0 = ['The user did not complete the purchase',
                    'The browsing took place in Nov',
                    'Operating Systems is 2',
                    'Browser is 2',
                    'Region is 1',
                    'Traffic Type is 2',
                    'Visitor Type is Returning_Visitor',
                    'Session is not on a weekend',
                    ]
            sen1 = ['The user completed the purchase',
                    'The browsing took place in Nov',
                    'Operating Systems is 2',
                    'Browser is 2',
                    'Region is 1',
                    'Traffic Type is 2',
                    'Visitor Type is Returning_Visitor',
                    'Session is not on a weekend',
                    ]
            sen_list.append(sen0)
            sen_list.append(sen1)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'False': cls_heads[0], 'True': cls_heads[1]}

        elif dataset_name == "buddy":
            for i in range(3):
                sen = [f'pet category is {i}',
                       # 'condition is 0.0',
                       'color type is Black',
                       'X1 is 0',
                       'X2 is 1',
                       'breed category is 2'
                       ]
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'0': cls_heads[0], '1': cls_heads[1], '2': cls_heads[2]}

        elif dataset_name == "obesity":
            for i in range(7):
                NObeyesdad = dataset.info['NObeyesdad'][str(i)]
                # sen = [f'obesity level is {NObeyesdad}', 'transportation method is Automobile', 'take extra calories? Sometimes']
                # sen = [f'obesity level is {NObeyesdad}', 'transportation method is Public_Transportation']
                # sen = [f'obesity level is {NObeyesdad}', 'gender is Male']
                sen = [f'obesity level is {NObeyesdad}', ]
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'Insufficient_Weight': cls_heads[0],
                             'Normal_Weight': cls_heads[1],
                             'Obesity_Type_I': cls_heads[2],
                             'Obesity_Type_II': cls_heads[3],
                             'Obesity_Type_III': cls_heads[4],
                             'Overweight_Level_I': cls_heads[5],
                             'Overweight_Level_II': cls_heads[6]
                             }

        elif dataset_name == "magic":
            sen0 = ['gamma"']
            sen1 = ['hadron']
            sen_list.append(sen0)
            sen_list.append(sen1)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'g': cls_heads[0], 'h': cls_heads[1]}

        elif dataset_name == "bean":
            for i in range(7):
                Class = dataset.info['Class'][str(i)]
                sen = [f'{Class}']
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'BARBUNYA': cls_heads[0],
                             'BOMBAY': cls_heads[1],
                             'CALI': cls_heads[2],
                             'DERMASON': cls_heads[3],
                             'HOROZ': cls_heads[4],
                             'SEKER': cls_heads[5],
                             'SIRA': cls_heads[6]
                             }

        elif dataset_name == "page":
            for i in range(5):
                Class = dataset.info['Class'][str(i+1)]
                sen = [f'{Class}']
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'1': cls_heads[0],
                             '2': cls_heads[1],
                             '3': cls_heads[2],
                             '4': cls_heads[3],
                             '5': cls_heads[4],
                             }

        elif dataset_name == "churn":
            sen0 = ['the customer is retained',
                    # 'gender is Male',
                    # 'country is France',
                    # 'The customer is using 1 bank products',
                    # 'the customer holds a credit card',
                    # 'the customer is not an active member',
                    ]
            sen1 = ['the customer closed account',
                    # 'gender is Male',
                    # 'country is France',
                    # 'The customer is using 1 bank products',
                    # 'the customer holds a credit card',
                    # 'the customer is not an active member',
                    ]
            sen_list.append(sen0)
            sen_list.append(sen1)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'0': cls_heads[0], '1': cls_heads[1]}

        elif dataset_name == "abalone":
            sen0 = ['The sex of abalone is female']
            sen1 = ['The sex of abalone is male']
            sen2 = ['The sex of abalone is infant']
            sen_list.append(sen0)
            sen_list.append(sen1)
            sen_list.append(sen2)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'F': cls_heads[0], 'M': cls_heads[1], 'I': cls_heads[2]}

        elif dataset_name == "bike":
            for w in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
                sen = [f'Today is {w}',
                       'season is summer',
                       'year is 2011',
                       'month is July',
                       "it's about 22 o'clock",
                       'today is not holiday',
                       'today is workingday',
                       "weather is Clear"
                       ]
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'Sunday': cls_heads[0],
                             'Monday': cls_heads[1],
                             'Tuesday': cls_heads[2],
                             'Wednesday': cls_heads[3],
                             'Thursday': cls_heads[4],
                             'Friday': cls_heads[5],
                             'Saturday': cls_heads[6]
                             }

        elif dataset_name == "insurance":
            for r in ['northeast', 'northwest', 'southeast', 'southwest']:
                sen = [f'region is {r}',
                       # "sex is female",
                       "The number of children is 0",
                       # "smoker? no"
                       ]
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {"northeast": cls_heads[0],
                             "northwest": cls_heads[1],
                             "southeast": cls_heads[2],
                             "southwest": cls_heads[3],
                             }

        elif dataset_name == "productivity":
            for w in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']:
                sen = [f'today is {w}',
                       "quarter is Quarter1",
                       "department is sweing"
                       ]
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'Sunday': cls_heads[0],
                             'Monday': cls_heads[1],
                             'Tuesday': cls_heads[2],
                             'Wednesday': cls_heads[3],
                             'Thursday': cls_heads[4],
                             'Saturday': cls_heads[5]
                             }

        elif dataset_name == "covertype":
            for i in range(7):
                cover_type = dataset.info['cover_type'][str(i)]
                # sen = [f'Forest Cover Type is {cover_type}']
                sen = [f'Forest Cover Type is {cover_type}', 'The flora is located in the Rawah Wilderness Area']
                sen_list.append(sen)

            cls_heads = []
            for i in range(len(sen_list)):
                token = berttokenizer(sen_list[i], padding=True, return_tensors="pt")
                input_ids = token['input_ids']
                attention_mask = token['attention_mask']
                hidden_rep, cls_head = bertmodel(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                cls_heads.append(cls_head)
            cls_heads_dic = {'0': cls_heads[0], '1': cls_heads[1], '2': cls_heads[2], '3': cls_heads[3],
                             '4': cls_heads[4], '5': cls_heads[5], '6': cls_heads[6]}

        else:
            raise ValueError("worng real_data_path!")


        all_cls_head = []
        if raw_config['Transform']['y_policy'] == 'Ordinal':
            [categories] = dataset.y_transformer.categories_
            print(categories)
            for y in label:
                c = categories[int(y)]
                all_cls_head.append(cls_heads_dic[c])

        elif raw_config['Transform']['y_policy'] == 'None':
            for y in label:
                all_cls_head.append(cls_heads_dic[str(int(y))])

        return torch.stack(all_cls_head)


if __name__ == '__main__':
    from lib import util


    raw_config = util.load_config("D:\Study\自学\表格数据生成/v11\exp/shopper\CoTable\config.toml")
    X_num, X_cat, y = read_pure_data(raw_config['real_data_path'], raw_config, split='train')
    X_cat = X_cat if X_cat is not None else np.empty((X_num.shape[0], 0))
    if y.ndim == 1:
        y = y[:, None]
    X_num_columns = raw_config['X_num_columns']
    X_cat_columns = raw_config['X_cat_columns']
    y_column = raw_config['y_column']
    df = pd.DataFrame(np.concatenate((X_num, X_cat, y), axis=1), columns=X_num_columns + X_cat_columns + y_column)
    print(df)

    # all_last_hidden_states = []
    all_pooler_outputs = []

    # 对所有行进行处理，并将编码的结果存储
    for _, row in df.iterrows():
        sentences = row_to_sentences(row, raw_config, with_label=True)
        print(sentences)
        if _ == 15:
            break