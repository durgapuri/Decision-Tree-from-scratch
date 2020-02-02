#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
import sys
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class DecisionTree:
    column_names = []
    decision_tree={}
    
    def fill_missing_values(self,train_data_frm):
#         print("colll ", train_data_frm.columns)
        for col_index in range(len(train_data_frm.columns)):
            cur_col_name = self.column_names[col_index]
#             print("cur_col_name ", cur_col_name)
#             print("train_data_frm[cur_col_name] ", train_data_frm[cur_col_name])
            if self.get_type(col_index):
                cur_col_name = self.column_names[col_index]
                train_data_frm[cur_col_name].fillna(train_data_frm[cur_col_name].mean(), inplace=True)
            else:
                cur_col_name = self.column_names[col_index]
                train_data_frm[cur_col_name].fillna(train_data_frm[cur_col_name].mode()[0], inplace=True)
        return train_data_frm
    
    def drop_columns(self,train_data_frm):
        train_data_frm = train_data_frm.drop('Id', axis=1)
        train_data_frm = train_data_frm.drop('Alley', axis=1)
        train_data_frm = train_data_frm.drop('PoolQC', axis=1)
        train_data_frm = train_data_frm.drop('Fence', axis=1)
        train_data_frm = train_data_frm.drop('MiscFeature', axis=1)
        return train_data_frm
    
    def prepare_data(self,train_data_frm):
        train_data_frm = self.drop_columns(train_data_frm)
        self.column_names = list(train_data_frm.columns)
        train_data_frm = self.fill_missing_values(train_data_frm)
        return train_data_frm
    
    def train_validation_split(self,data_frm,validation_data_size):
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(data_frm))

        indices=data_frm.index.tolist()

        valid_indices=random.sample(indices, validation_data_size)
        valid_datafrm=data_frm.loc[valid_indices]

        train_datafrm=data_frm.drop(valid_indices)

        return train_datafrm , valid_datafrm
    
    
    def check_pure(self,train_data):
        num_rows , num_cols = train_data.shape
        check_col = train_data[:,num_cols-1]
        unique_values = np.unique(check_col)
        if len(unique_values)==1:
            return True
        return False
    
    def get_type(self,index):
#         print(index)
#         print(self.column_names)
        col_name = self.column_names[index]
        continous_feat =['YrSold','MoSold','MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','OpenPorchSF','WoodDeckSF','GarageArea','GarageCars','GarageYrBlt','Fireplaces','TotRmsAbvGrd','Kitchen','Bedroom','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
        if col_name in continous_feat:
            return True
        return False
    
    def cal_current_mean_square(self,num_of_rows_below_split, num_of_rows_above_split,mean_square_below_values,
                            mean_square_above_values,num_of_rows):
    
        current_mean_square = (num_of_rows_below_split/num_of_rows)*mean_square_below_values + (num_of_rows_above_split/num_of_rows)*mean_square_above_values
        return current_mean_square
    
    def cal_mean_square(self,splitted_data):
    #     print("in cal_mean_square")
        num_of_rows, num_of_cols = splitted_data.shape
        price_values = splitted_data[:,num_of_cols-1]
        overall_mean = price_values.mean()
    #     print(price_values)
        mean_square = np.square(np.subtract(price_values,overall_mean)).mean()
        return mean_square
    
    def get_best_split(self,column_wise_potential_splits, train_data):
    #     print("in get_best_split")
        num_of_rows ,num_of_cols = train_data.shape
        best_split_col = None
        best_split_val = None
        overall_mean_square = sys.maxsize
    #     print(num_of_cols-1)
        overall_mean = train_data[:,num_of_cols-1].mean()
    #     print(overall_mean)
        for col_index,split_list in column_wise_potential_splits.items():
    #         print(col_index)
            col_values = train_data[:,col_index]
    #         print(column_names[col_index])
            for split_val in split_list:
                values_below_split_val = []
                values_above_split_val = []
                if self.get_type(col_index):
                    values_below_split_val = train_data[col_values <= split_val]
                    values_above_split_val = train_data[col_values > split_val]
    #                 print("continous")
                else:
                    values_below_split_val = train_data[col_values == split_val]
                    values_above_split_val = train_data[col_values != split_val]
    #                 print("cat")

                mean_square_below_values=0
                if len(values_below_split_val)!=0:
                    mean_square_below_values = self.cal_mean_square(values_below_split_val)

                mean_square_above_values=0
                if len(values_above_split_val)!=0:
                    mean_square_above_values = self.cal_mean_square(values_above_split_val)

                num_of_rows_below_split ,num_of_cols_below_split = values_below_split_val.shape
                num_of_rows_above_split ,num_of_cols_above_split = values_above_split_val.shape
                current_mean_square = self.cal_current_mean_square(num_of_rows_below_split, num_of_rows_above_split,
                                                              mean_square_below_values, mean_square_above_values,
                                                              num_of_rows)
                if current_mean_square < overall_mean_square:
                    overall_mean_square = current_mean_square
                    best_split_col = col_index
                    best_split_val = split_val
    #         print(overall_mean_square)
    #             print("below ",mean_square_below_values)
    #             print("above ",mean_square_below_values)
        return best_split_col, best_split_val


    def get_column_wise_potential_splits(self,train_data):
        column_wise_potential_splits = {}
        num_of_rows , num_of_cols = train_data.shape
    #     print(num_of_rows , num_of_cols)
        for index in range(num_of_cols-1):

                column_values = train_data[:,index]
                column_unique_values = np.unique(column_values)

                if(self.get_type(index)):
                    li=[]
                    for i in range(1,len(column_unique_values)):
                        li.append(((column_unique_values[i-1])+(column_unique_values[i]))/2)
                    column_wise_potential_splits[index]=li
                else:
                    column_wise_potential_splits[index] = column_unique_values.tolist()

        return column_wise_potential_splits   

    def decision_tree_algo(self,train_data, current_level=0, max_depth=4):
        num_rows, num_cols = train_data.shape

        if self.check_pure(train_data) or num_rows<=2 or current_level==max_depth:
            return train_data[:,num_cols-1].mean()

        else:
            current_level+=1
            column_wise_potential_splits = self.get_column_wise_potential_splits(train_data)
            best_col , best_split_val = self.get_best_split(column_wise_potential_splits, train_data)
    #         print(best_col, best_split_val)
            col_values = train_data[:,best_col]
            data_left_tree=[]
            data_right_tree=[]
            if self.get_type(best_col):
                data_left_tree = train_data[col_values <= best_split_val]
                data_right_tree = train_data[col_values > best_split_val]
    #                 print("continous")
            else:
                data_left_tree = train_data[col_values == best_split_val]
                data_right_tree = train_data[col_values != best_split_val]
    #                 print("cat")

            split_feature = self.column_names[best_col]
            qu = "{} {}".format(best_col, best_split_val)
            subtree = {qu: []}

            left_tree = self.decision_tree_algo(data_left_tree, current_level, max_depth)
            right_tree = self.decision_tree_algo(data_right_tree, current_level, max_depth)

            if left_tree==right_tree:
                subtree=left_tree
            else:
                subtree[qu].append(left_tree)
                subtree[qu].append(right_tree)

            return subtree
        
    def get_price_from_tree(self,test_sample,tree):
        list_qu = list(tree.keys())
        ques = list_qu[0]
        split_index , split_value = ques.split(" ")

        predicted_price = None
        if self.get_type(int(split_index)):
#             print("in")
            if test_sample[int(split_index)] <= float(split_value):
                predicted_price = tree[ques][0]
            else:
                predicted_price = tree[ques][1]
        else:
            if test_sample[int(split_index)] == split_value:
                predicted_price = tree[ques][0]
            else:
                predicted_price = tree[ques][1]

        if isinstance(predicted_price, dict):
            return self.get_price_from_tree(test_sample,predicted_price)
        else:
            return predicted_price
    
    def get_predicted_price(self, test_data):
        predicted_price=[]
        for test_sample in test_data:
            current_predicted_price = self.get_price_from_tree(test_sample,self.decision_tree)
            predicted_price.append(current_predicted_price)
        return predicted_price
   
    def check_validation(self, train_data_frm, validation_data_size):
        train_data_frm , validation_data_frm = self.train_validation_split(train_data_frm, validation_data_size)
        train_data = train_data_frm.values
        self.decision_tree = self.decision_tree_algo(train_data, max_depth=3)
        print(self.decision_tree)
        
        validation_data_labels = validation_data_frm.iloc[:,-1].to_frame().values.tolist()
        validation_data_frm = validation_data_frm.drop([validation_data_frm.columns[-1]],  axis='columns')
        validation_data = validation_data_frm.values
        predicted_price = self.get_predicted_price(validation_data)
        
        print(mean_squared_error(validation_data_labels, predicted_price))
        print(r2_score(validation_data_labels,predicted_price))
        
    
    
    def train(self,train_path):
        train_data_frm = pd.read_csv(train_path)
        # print(train_data_frm.info())
        # print(train_data_frm.shape)
        train_data_frm = self.prepare_data(train_data_frm)
#         self.check_validation(train_data_frm, validation_data_size = 10)
        # # print(train_data_frm.info())
        # print(train_data_frm.shape)
        # print(train_data_frm.info())
        # train_data_frm.shape
#         validation_data_size = 10;
#         random.seed(0)
        # train_data , valid_data = train_validation_split(train_data_frm, validation_data_size)
        train_data = train_data_frm.values
        self.decision_tree = self.decision_tree_algo(train_data, max_depth=5)
#         print(self.decision_tree)
        
    def predict(self,test_path):
        test_data_frm = pd.read_csv(test_path)
        test_data_frm = self.prepare_data(test_data_frm)
        # test_data_frm.info()
        test_data = test_data_frm.values
        # test_data = test_data[:3]
        predicted_price = self.get_predicted_price(test_data)
        return predicted_price
    


# In[97]:


# # import DecisionTree as dtree
# dtree_regressor = DecisionTree()
# dtree_regressor.train('/home/jyoti/Documents/SMAI/assign1/q3/train.csv')
# predictions = dtree_regressor.predict('/home/jyoti/Documents/SMAI/assign1/q3/test.csv')
# # print(predictions)
# test_labels = list()
# with open('/home/jyoti/Documents/SMAI/assign1/q3/test_labels.csv') as f:
#   for line in f:
#     test_labels.append(float(line.split(',')[1]))
# print(mean_squared_error(test_labels, predictions))
# print(r2_score(test_labels, predictions))


# In[ ]:




