# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:56:42 2019

@author: PARTHO PROTIM DEY
"""

from keras.models import load_model
from news_data import News_Data
#from connectSql import ConnectSql
#import math
import pandas as pd

def __normalize_headline(row):
        result = row.lower()
        # Delete useless character strings
        result = result.replace('...', '').replace('- ', '')
        
        whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')
        result = ''.join(filter(whitelist.__contains__, result))
        return result.strip()

try:
    stock="ABBANK"
    filename = './old_data/combined_'+stock+'_news.csv'        
    data = News_Data(filename,top_words=2000)
    test_sentence_str = __normalize_headline('Price Limit Open')            
    test_sentence = data.test_sentence(test_sentence_str)
    for d in ['today', 'tomorrow', 'day_after_tomorrow']:
        model = load_model('./model/combined_'+stock+'_news_'+d+'_.hdf5') 
        pre=model.predict(test_sentence, verbose=0)
        print (pre[0][0]*100)
    
    
# =============================================================================
#     #This sction was done for all days prediction with Sql server database
#      stock="ABBANK"
#     minDateForLoad="2010-10-28 18:00:00"
#     stockList=pd.DataFrame({'QuoteCode': ["ABBANK"]}) 
# #    stock="ABBANK"
#     for index, row in stockList.iterrows():
#         stock=row['QuoteCode']        
#         if stock=='00DSEX' or stock=='00DSES' or stock=='00DS30' or stock=='1JANATAMF': 
#             continue
#         print('Prediction start for '+ row['QuoteCode'])
#         df_news=ConnectSql().loadNewsFrmDate(stock,minDateForLoad)
#         df_news_filtered = df_news[~df_news['Content'].str.startswith('(Continuation')]
#         
#     #    stock=row['Scrip']
#         filename = './data/combined_'+stock+'_news.csv'        
#         data = News_Data(filename,top_words=2000)
#         dtLst=[]
#         todayLst=[]
#         tomorrowLst=[]
#         day_after_tomorrowLst=[]
#         actualValTodayLst=[]
#         actualValTomorrowLst=[]
#         actualValDATomorrowLst=[]
#         for d in ['today', 'tomorrow', 'day_after_tomorrow']:            
#             model = load_model('./model/combined_'+stock+'_news_'+d+'_.hdf5') 
#             for idx, ro in df_news_filtered.iterrows():    
#     #            if(index==5): break            
#                 test_sentence_str = __normalize_headline(ro['Title'])            
#                 test_sentence = data.test_sentence(test_sentence_str)            
#                 pre=model.predict(test_sentence)
#                 i, h = math.modf(pre[0][0]*100)
#                 
# #                print (row['PublishDate'].strftime('%Y-%m-%d') +'-'+ d +'-'+ str(i*100))
#                 if(d=='today'):
#                     dtLst.append(ro['PublishDate'].strftime('%Y-%m-%d'))
#                     todayLst.append(i*100)
#                     actualValTodayLst.append(pre[0][0]*100)
#                 elif(d=='tomorrow'):
#                     tomorrowLst.append(i*100)
#                     actualValTomorrowLst.append(pre[0][0]*100)
#                 else:
#                     day_after_tomorrowLst.append(i*100)
#                     actualValDATomorrowLst.append(pre[0][0]*100)
#          
#         cols = ['Date', 'Today', 'Tomorrow', 'DATomorrow', 'CreatedBy']
#         df_rslt = pd.DataFrame()
#         df_rslt['Date']=dtLst
#         df_rslt['Today']=todayLst
#         df_rslt['Tomorrow']=tomorrowLst
#         df_rslt['DATomorrow']=day_after_tomorrowLst
#         df_rslt['ActualValToday']=actualValTodayLst
#         df_rslt['ActualValTomorrow']=actualValTomorrowLst
#         df_rslt['ActualValDATomorrow']=actualValDATomorrowLst
#         df_rslt=df_rslt.groupby(['Date'], as_index=False).mean()
#         df_rslt['Scrip']=stock
#         df_rslt['CreatedBy']="PPD"
#         ConnectSql().SaveStockPredict(df_rslt)
#         print('Prediction end for '+ row['QuoteCode'])
# =============================================================================
except Exception as ex:
    print(ex)
