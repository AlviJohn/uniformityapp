import streamlit as st
import numpy as np
import pandas as pd
from chart_studio import plotly as py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import product
from datetime import datetime
import time
from PIL import Image
from azure.storage.blob import BlobServiceClient,ContentSettings, ContainerClient
import os
from datetime import date, timedelta


######Setting the Basics for the Page
st.set_page_config(page_title="UniformityApp", page_icon="muscleman.jpg", layout="wide", initial_sidebar_state="auto")
st.title('Uniformity Dashboard')


#########################################Helper Functions###########################################
##############################Reading the Data and basic Processing of datetime
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def process_input(df):
    df = df[['TireType','gt_dom','curing_dom','rej_param','tbmref','curing_machine','BARCODE','RFVCW', 'CONICITY', 'Static']]    
    df['gt_dom'] = pd.to_datetime(df['gt_dom'])
    df['curing_dom']=pd.to_datetime(df['curing_dom'])    
    df=df.sort_values(by='gt_dom')
    
    df = df.dropna(subset = ['gt_dom', 'curing_dom'])

    ####Adjusting for GT Dom Time
    df['Date'] = df['gt_dom'].dt.date

    df['hour_indicator'] =[ -1 if x<7 else 0 for x in df['gt_dom'].dt.hour]
    df['Date'] = df['Date'] + df['hour_indicator'] * pd.to_timedelta("1day")

   
    df['Curing_Date']=df['curing_dom'].dt.date
    df['curing_hour_indicator'] =[ -1 if x<7 else 0 for x in df['curing_dom'].dt.hour]
    df['Curing_Date'] = df['Curing_Date'] + df['curing_hour_indicator'] * pd.to_timedelta("1day")
       
    #df.to_csv('test1.csv')    
    df['rej_param'] = df['rej_param'].fillna('No Rejection')
    df['Rejection_Reason'] = df['rej_param'].astype('str')
    df['curing_machine'] = df['curing_machine'].astype('str').replace('\.0', '', regex=True)
    df['Rejection_Reason'] = np.where(df['Rejection_Reason'].isin(['RFPP','RFH1','RFH2','LFPP']), 'RFPP',df['Rejection_Reason'])
    #df['Rejection_Reason'] =df.replace(np.nan, 'No Rejection', regex=True)
    df['uniformity_status'] = [0 if x == 'No Rejection' else 1 for x in df.Rejection_Reason]
    
    return df

#######################################Function to Create Yield Charts
#@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def Yield_charts(df):
    df['Yield_Status'] = [1 if x == 'No Rejection' else 0 for x in df.Rejection_Reason]
    return df
                            


#######################################Function to Create daily stack charts 
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def daily_charts(df):
    

    #########For individual Cavity
    df_stackchart = pd.DataFrame(df.groupby(['Curing_Date','curing_machine'])['Rejection_Reason'].
                             value_counts(normalize=False)).rename(columns ={'Rejection_Reason':'num_tyres'})
    df_stackchart = df_stackchart.reset_index()

    df_total_Yield = df.groupby(['Curing_Date','curing_machine']).\
                    agg({'BARCODE':'count'}).\
                    reset_index().\
                    rename(columns={'BARCODE':'Total Tyres'})                                     
    df_stackchart =df_stackchart.merge(df_total_Yield, on=['Curing_Date','curing_machine'])    
    df_stackchart = df_stackchart.loc[~(df_stackchart['Rejection_Reason']=='No Rejection'),:]

    
    #########For Overall
    df_stackchart_overall = pd.DataFrame(df.groupby(['Curing_Date'])['Rejection_Reason'].
                             value_counts(normalize=False)).rename(columns ={'Rejection_Reason':'num_tyres'})
    
    df_stackchart_overall = df_stackchart_overall.reset_index()  
    df_stackchart_overall = df_stackchart_overall.loc[~(df_stackchart_overall['Rejection_Reason']=='No Rejection'),:]
    
    df_total_Yield_overall = df.groupby(['Curing_Date']).\
                    agg({'BARCODE':'count'}).\
                    reset_index().\
                    rename(columns={'BARCODE':'Total Tyres'})                  
    df_stackchart_overall =df_stackchart_overall.merge(df_total_Yield_overall, on=['Curing_Date'])
    df_stackchart_overall['curing_machine'] = 'Overall'

    df_stackchart_combined = pd.concat([df_stackchart,df_stackchart_overall])
   
#    df_stackchart_combined.to_csv('df_stackchart_combined.csv')
    return df_stackchart_combined


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def data_paramchart(df):
    df_temp = df[['Date','RFVCW','CONICITY','Static','BARCODE']]
    df_temp = df_temp.groupby(['Date']).\
                    agg({'BARCODE':'count','RFVCW':'sum','CONICITY':'sum','Static':'sum'  }).\
                    reset_index().\
                    rename(columns={'BARCODE':'Total Tyres'}) 
    
    df_temp= df_temp.sort_values(by=['Date'],axis=0) 
    return df_temp


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def data_paramchart(df):
    df_temp = df[['Date','RFVCW','CONICITY','Static','BARCODE']]
    df_temp = df_temp.groupby(['Date']).\
                    agg({'BARCODE':'count','RFVCW':'sum','CONICITY':'sum','Static':'sum'  }).\
                    reset_index().\
                    rename(columns={'BARCODE':'Total Tyres'}) 
    
    df_temp= df_temp.sort_values(by=['Date'],axis=0) 
    return df_temp



@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def cloud_datafetch():
    ###Connection with Azure Container
    source_key = 'wqzYUdetZd6ZkgSQIQsIDnU5n4OolDudleEsEPLClMdvxR8u3aZLOXvRpSI1oKzILo8kx3vnUlaWajxGZl5HbQ=='
    source_account_name = 'uniformity'
    block_blob_service = BlobServiceClient(account_url=f'https://{source_account_name}.blob.core.windows.net/', credential=source_key)
    client = block_blob_service.get_container_client('uniformityanalysis')

    print('connection established')
    ###File name 
    filename = 'TUO-DATA.csv'

    ### last 2 months date
    start_date = date.today()
    end_date = start_date - timedelta(days=60) #as if now given last 2 days / in case of month timedelta(months=2)
    daterange = pd.date_range(end_date, start_date)

    lists=[]
    tuocols= ['TireType','gt_dom','curing_dom','rej_param','tbmref','curing_machine','BARCODE','RFVCW', 'CONICITY', 'Static']
    ### fetching files ending with TUO-DAA.csv for last 2 months
    for date_ in daterange:
        blob_list = client.list_blobs( name_starts_with="Raw-Data/" + date_.strftime('%m-%d-%Y') + '/')
        for blob in blob_list:
            text=blob.name
            if text.endswith(filename):
                lists.append(text)
    st.write('Fetched file names')

    final_df = pd.DataFrame()
    ### read files as df and save it as csv files in local machine
    for blobs in lists:
        a=blobs.replace(" ", "%20")
        df= pd.read_csv('https://uniformity.blob.core.windows.net/uniformityanalysis/{file_}'.format(file_=a),low_memory = True,usecols =tuocols)
        final_df = final_df.append(df, ignore_index=True)

    return final_df



data_option = st.radio("Please select the Data Option",('Cloud_Data_Ingestion', 'Manual_Upload'))

if data_option =='Manual_Upload':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file,sheet_name='Sheet1')
if data_option =='Cloud_Data_Ingestion':
    uploaded_file = 'Cloud_Data_Ingestion'
    df = cloud_datafetch()
    #st.write(df.head())
    st.write('Fetched Last 2 months of TUO Data')



if uploaded_file is not None:
    
    df= process_input(df)

    ###########################################Selections#########################################
    st.sidebar.title('Dashboard Filters')
    st.sidebar.text('Filters selected will be applied to \nonly Daily Yield,Daily Rejection & \nDaily Parameter Plots')

    ###Selecting SKU
    SKU_choices = df['TireType'].unique().tolist()
    
    ###For Display Purpose only
    if data_option =='Cloud_Data_Ingestion':
        SKU_choices.insert(0,"FT16350681")
    
    SKU_make_choice = st.sidebar.selectbox('Select SKU', SKU_choices)
    SKU_make_choice = SKU_make_choice
    df = df.loc[((df['TireType'] ==SKU_make_choice))]


    ###Selecting the TBM 
    TBM_choices = df['tbmref'].unique().tolist()

    ###For Display Purpose only
    if data_option =='Cloud_Data_Ingestion':
        TBM_choices.insert(0,"S19")

    TBM_make_choice = st.sidebar.selectbox('Select TBM', TBM_choices)
    


    TBM_make_choice = TBM_make_choice
    ###########Selecting Cavity

    cavity_choices = df['curing_machine'].unique().tolist()
    cavity_choices_V2 =cavity_choices
    cavity_choices_V2.insert(0,"ALL")
    cavity_make_choice = st.sidebar.multiselect("Select one or more Cavities:",cavity_choices_V2,'ALL')

    if "ALL" in cavity_make_choice:
        cavity_make_choice_final = cavity_choices
    else:
        cavity_make_choice_final = cavity_make_choice


    df_temp=df
    df_temp=df.loc[((df['tbmref'] ==TBM_make_choice)) & (df['curing_machine'].isin(cavity_make_choice_final))]
    df_Yield = Yield_charts(df_temp)
    df_stackchart = daily_charts(df_temp)
    df_paramchart = data_paramchart(df_temp)


#######################################Functions for processing the charts#############################
#############################################################################


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def daily_chart1_Processing(df_Yield, MA_make_choice_chart1,rolling_val):
    
    df_Yield = pd.DataFrame(df_Yield.groupby(['Date']).
                            agg({'Yield_Status':'sum','BARCODE':'count'}).\
                            reset_index().\
                            rename(columns={'Yield_Status':'approved_tyres','BARCODE':'Total_Tyres'})) 
    
    df_Yield['Non_uniform_tyres' + MA_make_choice_chart1] = df_Yield['approved_tyres'].transform(lambda x: x.rolling(rolling_val).sum())   
    df_Yield['Total_Tyres' + MA_make_choice_chart1] = df_Yield['Total_Tyres'].transform(lambda x: x.rolling(rolling_val).sum())                                                                                                          
    df_Yield['Yield' + MA_make_choice_chart1] = round( (df_Yield['Non_uniform_tyres' + MA_make_choice_chart1]/df_Yield['Total_Tyres' + MA_make_choice_chart1]) * 100,2)    


    ###################Rejection
    df_rejection = pd.DataFrame(df.groupby(['Date'])['Rejection_Reason'].
                             value_counts(normalize=False)).rename(columns ={'Rejection_Reason':'num_tyres'})
    df_rejection = df_rejection.reset_index(drop=False)  
    df_rejection = df_rejection.loc[~(df_rejection['Rejection_Reason']=='No Rejection'),:]  
    
    
    
    combs = pd.DataFrame(list(product(df_rejection['Date'].unique(), df_rejection['Rejection_Reason'].unique())), 
                         columns=['Date', 'Rejection_Reason'])     
    df_rejection = df_rejection.merge(combs,how = 'right').fillna(0)
    
    
    df_rejection =df_rejection.merge(df_Yield[['Date','Total_Tyres']], on=['Date'])
    df_rejection= df_rejection.sort_values(by=['Rejection_Reason','Date'],axis=0) 

    
    df_rejection['rejected_tyres' + MA_make_choice_chart1] = df_rejection.groupby(['Rejection_Reason'])['num_tyres'].\
                                                        transform(lambda x: x.rolling(rolling_val).sum())  
                                                        
    df_rejection['Total_Tyres' + MA_make_choice_chart1]  = df_rejection.groupby(['Rejection_Reason'])['Total_Tyres'].\
                                                        transform(lambda x: x.rolling(rolling_val).sum()) 
    
    df_rejection['Rejection' + MA_make_choice_chart1] = round((df_rejection['rejected_tyres' + MA_make_choice_chart1]/df_rejection['Total_Tyres' + MA_make_choice_chart1]) * 100,2)  
    df_rejection['Rejection_Reason']= pd.Categorical(df_rejection.Rejection_Reason,categories=["RFPP","Imbalance","CONICITY","Runout", "Bulge","Dent","Overall"]) 
    df_rejection= df_rejection.sort_values(by=['Rejection_Reason','Date'],axis=0) 

  
    
    #df_rejection = '12'
    return df_Yield,df_rejection

    
   

#############Chart 3-Rejection_MA_chart3_Processing
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def Rejection_MA_chart3_Processing(df_stackchart_subset, MA_make_choice,cavity_make_choice):
       #####If Individual Cavity
    if cavity_make_choice[0] != 'ALL':
        
        
        df_tyres = df_stackchart[['Curing_Date','curing_machine','Total Tyres']].drop_duplicates()
        df_stackchart_subset =df_stackchart_subset[['Curing_Date','curing_machine','Rejection_Reason','num_tyres']]
        
        df_stackchart_V2 = df_stackchart_subset.loc[df_stackchart_subset['curing_machine']==cavity_make_choice[0],:]             	  
        

        combs = pd.DataFrame(list(product(df_stackchart_V2['Curing_Date'].unique(), df_stackchart_V2['Rejection_Reason'].unique(),df_stackchart_V2['curing_machine'].unique())), 
                         columns=['Curing_Date', 'Rejection_Reason','curing_machine'])
         
        df_stackchart_V2 = df_stackchart_V2.merge(combs,how = 'right').fillna(0)
        df_stackchart_V2 = df_stackchart_V2.merge(df_tyres,how = 'left').fillna(0)

        
        df_stackchart_V2 = df_stackchart_V2.drop(['Rejection3 Day Moving Average','Rejection14 Day Moving Average','Rejection7 Day Moving Average','Rejection5 Day Moving Average','RejectionDaily'], axis=1, errors='ignore')               
        df_stackchart_V2['Rejection_Reason']= pd.Categorical(df_stackchart_V2.Rejection_Reason,categories=["RFPP","Imbalance","CONICITY","Runout", "Bulge","Dent","Overall"])
        df_stackchart_V2= df_stackchart_V2.sort_values(by=['curing_machine','Rejection_Reason','Curing_Date'],axis=0)
        
        df_stackchart_V2['rejected_tyres' + MA_make_choice] = df_stackchart_V2.groupby(['Rejection_Reason'])['num_tyres'].\
                                                        transform(lambda x: x.rolling(rolling_val).sum())  
                                                        
        df_stackchart_V2['Total_Tyres' + MA_make_choice]  = df_stackchart_V2.groupby(['Rejection_Reason'])['Total Tyres'].\
                                                        transform(lambda x: x.rolling(rolling_val).sum()) 


        df_stackchart_V2['Rejection' + MA_make_choice] =   (df_stackchart_V2['rejected_tyres' + MA_make_choice]/df_stackchart_V2['Total_Tyres' + MA_make_choice])
                                                                                                        
        df_stackchart_V2['Rejection' + MA_make_choice] = round(df_stackchart_V2['Rejection' + MA_make_choice]* 100,2)
        
        



    ########If All Cavities Selected
    if cavity_make_choice[0]    == 'ALL':
    
        
        df_tyres = df_stackchart[['Curing_Date','curing_machine','Total Tyres']].drop_duplicates()
        
        df_stackchart_V2 =df_stackchart_subset[['Curing_Date','curing_machine','Rejection_Reason','num_tyres']]                   
        combs = pd.DataFrame(list(product(df_stackchart_V2['Curing_Date'].unique(), df_stackchart_V2['Rejection_Reason'].unique(),df_stackchart_V2['curing_machine'].unique())), 
                         columns=['Curing_Date', 'Rejection_Reason','curing_machine'])

        df_stackchart_V2 = df_stackchart_V2.merge(combs,  how = 'right').fillna(0)
        df_stackchart_V2 = df_stackchart_V2.merge(df_tyres,  how = 'left').fillna(0)
    
        
        df_stackchart_V2 = df_stackchart_V2.drop(['Rejection3 Day Moving Average','Rejection14 Day Moving Average','Rejection7 Day Moving Average','Rejection5 Day Moving Average','RejectionDaily'], axis=1, errors='ignore')            
       
        df_stackchart_V2['Rejection_Reason']= pd.Categorical(df_stackchart_V2.Rejection_Reason,categories=["RFPP","Imbalance","CONICITY","Runout", "Bulge","Dent","Overall"])
        
        df_stackchart_V2= df_stackchart_V2.sort_values(by=['curing_machine','Rejection_Reason','Curing_Date'],axis=0)    
        
     
        df_stackchart_V2['rejected_tyres' + MA_make_choice] = df_stackchart_V2.groupby(['curing_machine','Rejection_Reason'])['num_tyres'].\
                                                        transform(lambda x: x.rolling(rolling_val).sum())  
                                                        
        df_stackchart_V2['Total_Tyres' + MA_make_choice]  = df_stackchart_V2.groupby(['curing_machine','Rejection_Reason'])['Total Tyres'].\
                                                        transform(lambda x: x.rolling(rolling_val).sum()) 


        df_stackchart_V2['Rejection' + MA_make_choice] =   (df_stackchart_V2['rejected_tyres' + MA_make_choice]/df_stackchart_V2['Total_Tyres' + MA_make_choice])
   

        df_stackchart_V2['Rejection' + MA_make_choice] = round(df_stackchart_V2['Rejection' + MA_make_choice]* 100,2) 
        df_stackchart_V2 = df_stackchart_V2.dropna()

        
    return df_stackchart_V2


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def Daily_Triggers(df_stackchart, max_date):
    
    df_tyres = df_stackchart[['Curing_Date','curing_machine','Total Tyres']].drop_duplicates() 
    
    df_stackchart_trigger = df_stackchart[['Curing_Date','curing_machine','Rejection_Reason', 'num_tyres']]
        ###Adding Total Rejections as a summary
    df_stackchart_trigger = df_stackchart_trigger.groupby(['Curing_Date','curing_machine'])\
                                                  .agg({'num_tyres':'sum'}).reset_index()\
                                                 .assign(Rejection_Reason='Total').append(df_stackchart_trigger)
   
   
    ####Data Operations to ensure that there all combinations are there
    combs = pd.DataFrame(list(product(df_stackchart_trigger['Curing_Date'].unique(), df_stackchart_trigger['Rejection_Reason'].unique(),df_stackchart_trigger['curing_machine'].unique())), 
                         columns=['Curing_Date', 'Rejection_Reason','curing_machine'])

    df_stackchart_trigger = df_stackchart_trigger.merge(combs,  how = 'right').fillna(0)
    df_stackchart_trigger = df_stackchart_trigger.merge(df_tyres,how='inner').fillna(0)

          
    df_stackchart_trigger = df_stackchart_trigger.loc[df_stackchart_trigger['Rejection_Reason'].isin(['RFPP','Imbalance','Total']),:]
    df_stackchart_trigger['Rejection_percent'] = round((df_stackchart_trigger['num_tyres']/df_stackchart_trigger['Total Tyres'])*100,2)

    #####Last day
    last_day = df_stackchart_trigger.loc[df_stackchart_trigger['Curing_Date'] ==max_date ,:].drop(['Curing_Date'], axis=1)
    

     
    #####Worst 3 Cavities
    last_day= last_day.sort_values(by=['Rejection_Reason','Rejection_percent'], ascending = [False, False],axis=0)
    
    top_df= last_day.groupby('Rejection_Reason').head(3)
    top_df= top_df[['curing_machine','Rejection_Reason','Rejection_percent','num_tyres','Total Tyres']].reset_index(drop=True)
    top_df= top_df.rename(columns ={'num_tyres':'tyres_rejected'})
    top_df['tyres_rejected']=top_df['tyres_rejected'].astype(int)
    


    
    return df_stackchart_trigger,top_df,last_day;


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def Daily_rejection_trends(df_stackchart_trigger, max_date,lookback_time):
    
    df_stackchart_trigger = df_stackchart_trigger.loc[(df_stackchart_trigger['Curing_Date'] >= (max_date - pd.to_timedelta(str(lookback_time) + "day"))) & (df_stackchart_trigger['Curing_Date'] <= (max_date)) ,:]


    ####Moving Averages
    df_stackchart_trigger= df_stackchart_trigger.sort_values(by=['curing_machine','Rejection_Reason','Curing_Date'],axis=0)
    
    
    df_stackchart_trigger['rejected_tyres_3DMA'] = df_stackchart_trigger.groupby(['curing_machine','Rejection_Reason'])['num_tyres'].\
                                                        transform(lambda x: x.rolling(3).sum())  
                                                        
    df_stackchart_trigger['Total_Tyres_3DMA']  = df_stackchart_trigger.groupby(['curing_machine','Rejection_Reason'])['Total Tyres'].\
                                                        transform(lambda x: x.rolling(3).sum()) 
    
    df_stackchart_trigger['Rejection_3DMA'] = round((df_stackchart_trigger['rejected_tyres_3DMA']/df_stackchart_trigger['Total_Tyres_3DMA'])*100,2)
    
    
    last_day = df_stackchart_trigger.loc[df_stackchart_trigger['Curing_Date'] ==max_date ,:].drop(['Curing_Date'], axis=1)
    ########Moving Average of all the days
    average_deviation = df_stackchart_trigger.loc[df_stackchart_trigger['Curing_Date'] < max_date ,:]\
                                            .groupby(['curing_machine','Rejection_Reason']).\
                                            agg({'Rejection_3DMA': ['mean', 'std']}).reset_index()
    
    average_deviation.columns=['curing_machine','Rejection_Reason','Rejection_avg_3DMA','Rejection_std_3DMA']

    trend_df = last_day.merge(average_deviation, on=['curing_machine','Rejection_Reason'])
    trend_df = trend_df[['curing_machine','Rejection_Reason','Rejection_3DMA','Rejection_avg_3DMA','Rejection_std_3DMA']]

  
    return df_stackchart_trigger,trend_df;
    
 
#####################################Chart 1 Daily Yield Chart######################################################################
if uploaded_file is not None:
    with st.expander('Daily Yield Chart'):
        
        start = time.time()
        MA_make_choice_chart1 = st.selectbox('Select the Level for Yeild Chart', 
                                      ('Daily','3 Day Moving Average', 
                                       '7 Day Moving Average', '5 Day Moving Average',
                                       '14 Day Moving Average'))

        
        if MA_make_choice_chart1=='3 Day Moving Average':
            rolling_val=3
        elif MA_make_choice_chart1=='Daily':
            rolling_val=1    
        elif MA_make_choice_chart1=='7 Day Moving Average':
            rolling_val=7
        elif MA_make_choice_chart1=='5 Day Moving Average':
            rolling_val=5
        elif MA_make_choice_chart1=='14 Day Moving Average':
            rolling_val=14
        


        df_Yield,df_rejection = daily_chart1_Processing(df_Yield, MA_make_choice_chart1,rolling_val)
        

        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(df_Yield['Date']),
                             y=list(df_Yield['Yield' + MA_make_choice_chart1]),name= 'Daily Yield',opacity=1,marker_line_color='#11457E'))
        
        
        fig.update_layout(
            title_text="TBM Yield Chart"
        )
        
         
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3day",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="7day",
                             step="day",
                             stepmode="todate"),
                        dict(count=14,
                             label="14day",
                             step="day",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        
        fig.update_layout(yaxis_range=[60,100])
        st.plotly_chart(fig, use_container_width=True)
        csv= df_Yield.to_csv().encode('utf-8')
        st.download_button(
       "Click to Download Data",
        csv,
       "TBM_Yield"+MA_make_choice_chart1+".csv",
       "text/csv",
       key='download-csv1'
       )

       
        
        
        fig1 = px.bar(df_rejection, x="Date", y='Rejection' + MA_make_choice_chart1 , color="Rejection_Reason",
                             text=[f'{i}%' for i in df_rejection['Rejection' + MA_make_choice_chart1]]) 
        fig1.layout.yaxis.tickformat = '0'
        fig1.update_layout(title_text= "TBM Rejection Chart")
        st.plotly_chart(fig1, use_container_width=True)
                
        st.write("% s seconds to run the code" % round((time.time() - start),3))      
       
        csv= df_rejection[['Date','Rejection_Reason','num_tyres','Total_Tyres','Rejection' + MA_make_choice_chart1]].to_csv().encode('utf-8')
        st.download_button(
       "Click to Download Data",
        csv,
       "TBM_"+MA_make_choice_chart1+"_Rejection.csv",
       "text/csv",
       key='download-csv2'
       )



    ###########################################Chart 2--Cavity Chart
    with st.expander('Cavity Comparison -Overall'):
        start = time.time()
    #########Calendar for Cacity Comparison
        min_date = df['Curing_Date'].min()
        max_date = df['Curing_Date'].max()
        #######Date Selection for Cavity
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start date', min_date)
        with col2:
            end_date = st.date_input('End date', max_date)
        if start_date > end_date:
            st.error('Error: End date must fall after start date.')
            


            
        df_cavity = pd.DataFrame(
                        df[(df['Curing_Date'] >= start_date) & (df['Curing_Date'] <= end_date)].\
                        groupby(['curing_machine'])['Rejection_Reason'].\
                                 value_counts(normalize=True)).\
                                 rename(columns ={'Rejection_Reason':'Rejection_percent'})
                                 
        df_total_tyres = df[(df['Curing_Date'] >= start_date) & (df['Curing_Date'] <= end_date)].groupby(['curing_machine']).\
                        agg({'BARCODE':'count'}).\
                        reset_index().\
                        rename(columns={'BARCODE':'Total_tyres'})                         
                                 
        
        df_cavity = df_cavity.reset_index()
        df_cavity = df_cavity.merge(df_total_tyres,on=["curing_machine"])    
        df_cavity = df_cavity.loc[~(df_cavity['Rejection_Reason']=='No Rejection'),:]
        df_cavity = df_cavity.dropna()    
        df_cavity['Rejection_percent'] = round(df_cavity['Rejection_percent'] * 100,2)    
        df_cavity['curing_machine'] = 'Cavity_' + df_cavity['curing_machine'].astype(str) 


        df_cavity['Rejection_Reason']= pd.Categorical(df_cavity.Rejection_Reason,categories=["RFPP","Imbalance","CONICITY","Runout", "Bulge","Dent","Overall"])
        df_cavity= df_cavity.sort_values(by=['Rejection_Reason'],axis=0)
        
        fig = px.bar(df_cavity, x="curing_machine", y="Rejection_percent", color="Rejection_Reason", text=[f'{i}%' for i in df_cavity['Rejection_percent']])
        fig.update_layout(title_text="Rejection per Cavity")
        fig.layout.yaxis.tickformat = '0'
        st.plotly_chart(fig, use_container_width=True)           
        
        
        fig2 = px.bar(df_cavity, x="curing_machine", y="Total_tyres",barmode='overlay', 
                       text=[f'{i}' for i in df_cavity['Total_tyres']])
        
        fig2.update_layout(title_text="Number of Tyres")
        
        st.plotly_chart(fig2, use_container_width=True)           
        
        
        st.write("% s seconds to run the code" % round((time.time() - start),3))
        df_cavity['start_date'] =  start_date   
        df_cavity['end_date'] =  end_date    
        csv= df_cavity.to_csv().encode('utf-8')
        st.download_button(
       "Click to Download Data",
        csv,
       "Cavity_OverallRejection.csv",
       "text/csv",
       key='download-csv3'
       )


    ######################################Chart 3######################################################################
    with st.expander('Cavity Rejection by Reason Moving Average Charts'):
        start = time.time()
        start_date = df['Curing_Date'].min()
        end_date = df['Curing_Date'].max()
        
        #######Date Selection for Cavity
        col1, col2 = st.columns(2)
        with col1:
            starting_date = st.date_input('Starting date', start_date)
        with col2:
            ending_date = st.date_input('Ending date', end_date)
        if starting_date > ending_date:
            st.error('Error: End date must fall after start date.')

            
        df_stackchart_subset = df_stackchart.loc[(df_stackchart['Curing_Date'] >= starting_date) & (df_stackchart['Curing_Date'] <= ending_date),:]

        MA_make_choice = st.selectbox('Select the Level for Rejection Values', 
                                      ('Daily','3 Day Moving Average', 
                                       '7 Day Moving Average', '5 Day Moving Average',
                                       '14 Day Moving Average'))

        
        if MA_make_choice=='3 Day Moving Average':
            rolling_val=3
        elif MA_make_choice=='Daily':
            rolling_val=1    
        elif MA_make_choice=='7 Day Moving Average':
            rolling_val=7
        elif MA_make_choice=='5 Day Moving Average':
            rolling_val=5
        elif MA_make_choice=='14 Day Moving Average':
            rolling_val=14

     

        df_stackchart_V2 =Rejection_MA_chart3_Processing(df_stackchart_subset, MA_make_choice,cavity_make_choice)

        
        
        if cavity_make_choice[0] != 'ALL':
            
            fig = px.bar(df_stackchart_V2, x="Curing_Date", y='Rejection' + MA_make_choice , color="Rejection_Reason",
                         text=[f'{i}%' for i in df_stackchart_V2['Rejection' + MA_make_choice]],category_orders={'Rejection_Reason':["Bulge","Dent","Runout","Conicity","Imbalance","RFPP"]})
            
            fig.update_layout(title_text="Rejection Moving Average (Each data point represents "+ MA_make_choice + " till that date)")
            fig.update_yaxes(title_text="Yield Percentage", secondary_y=True)   
            fig.layout.yaxis.tickformat = '0'
            st.plotly_chart(fig, use_container_width=True)
            
            df_stackchart_V2['start_date'] = starting_date
            df_stackchart_V2['end_date'] = ending_date
            csv= df_stackchart_V2[['Curing_Date','Rejection_Reason','num_tyres','Total Tyres', 'Rejection' + MA_make_choice,'start_date','end_date']].to_csv().encode('utf-8')

        if cavity_make_choice[0]    == 'ALL':
            unique_cavities = df_stackchart_V2['curing_machine'].unique().tolist()
            
            
            unique_cavities.remove("Overall")
            unique_cavities.insert(0, "Overall")

            for count, value in enumerate(unique_cavities):  
                temp = df_stackchart_V2.loc[df_stackchart_V2['curing_machine']==value,:]
                combs = pd.DataFrame(list(product(df_stackchart_V2['Curing_Date'].unique(), df_stackchart_V2['Rejection_Reason'].unique())), 
                             columns=['Curing_Date', 'Rejection_Reason'])
                temp = temp.merge(combs,  how = 'right').fillna(0)

                fig = px.bar(temp, x="Curing_Date", y='Rejection' + MA_make_choice , color="Rejection_Reason",
                             text=[f'{i}%' for i in temp['Rejection' + MA_make_choice]]) 
                fig.layout.yaxis.tickformat = '0'
                fig.update_layout(title_text= "Cavity-" +str(value)+ " - Each data point represents rejection"+ MA_make_choice + " till that date")
                st.plotly_chart(fig, use_container_width=True)
                
            
            df_stackchart_V2['start_date'] = starting_date
            df_stackchart_V2['end_date'] = ending_date
            
            
            csv= df_stackchart_V2[['Curing_Date','curing_machine','Rejection_Reason','num_tyres','Total Tyres', 'Rejection' + MA_make_choice,'start_date','end_date']].to_csv().encode('utf-8')


        st.write("% s seconds to run the code" % round((time.time() - start),3))      
        st.download_button(
       "Click to Download Data",
        csv,
       "Cavity_Rejection_"+MA_make_choice +".csv",
       "text/csv",
       key='download-csv4')

    ######################################Chart 3######################################################################
    with st.expander('Daily Uniformity Parameter Chart'):
        start = time.time()
        option = st.selectbox(
             'Select Uniformity Parameter',
             ('RFVCW', 'CONICITY', 'Static'))
        
        MA_make_choice_chart4 = st.selectbox('Select the level for uniformity parameter', 
                                      ('Daily','3 Day Moving Average', 
                                       '7 Day Moving Average', '5 Day Moving Average',
                                       '14 Day Moving Average'))

        
        if MA_make_choice_chart4=='3 Day Moving Average':
            rolling_val=3
        elif MA_make_choice_chart4=='Daily':
            rolling_val=1    
        elif MA_make_choice_chart4=='7 Day Moving Average':
            rolling_val=7
        elif MA_make_choice_chart4=='5 Day Moving Average':
            rolling_val=5
        elif MA_make_choice_chart4=='14 Day Moving Average':
            rolling_val=14
        
            
        df_paramchart_V1 = df_paramchart
        
        df_paramchart_V1[option +'_sum_'+ MA_make_choice_chart4] = df_paramchart_V1[option].\
                                                            transform(lambda x: x.rolling(rolling_val).sum())  
                                                            
        df_paramchart_V1['Totaltyres_'+ MA_make_choice_chart4] = df_paramchart_V1['Total Tyres'].\
                                                            transform(lambda x: x.rolling(rolling_val).sum())  
                                                            
        df_paramchart_V1[option+'_'+ MA_make_choice_chart4]  =   (df_paramchart_V1[option +'_sum_'+ MA_make_choice_chart4]/ df_paramchart_V1['Totaltyres_'+ MA_make_choice_chart4])                                         
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(df_paramchart_V1['Date']),
                             y=list(df_paramchart_V1[option +'_'+ MA_make_choice_chart4]),name= option,opacity=1,marker_line_color='#11457E'))
        
      
            
        fig.update_layout(
            title_text="Daily Uniformity Parameter"
        )
            
         
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=8,
                             label="1shift",
                             step="hour",
                             stepmode="backward"),
                        dict(count=1,
                             label="1day",
                             step="day",
                             stepmode="backward"),
                        dict(count=3,
                             label="3day",
                             step="day",
                             stepmode="backward"),
                        dict(count=7,
                             label="7day",
                             step="day",
                             stepmode="todate"),
                        dict(count=14,
                             label="14day",
                             step="day",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        csv= df_paramchart_V1[['Date','Total Tyres',option+'_'+ MA_make_choice_chart4]].to_csv().encode('utf-8')
        st.write("% s seconds to run the code" % round((time.time() - start),3))      
        st.download_button(
       "Click to Download Data",
        csv,
       "Uniformity_Parameter"+MA_make_choice_chart4+".csv",
       "text/csv",
       key='download-csv5'
       )

    #
    ################Code for Triggers
    #
    with st.expander('Daily Triggers'):
        start = time.time()
        if cavity_make_choice[0]=="ALL":        
                    
            
            start_data = df_stackchart['Curing_Date'].min()
            end_data = df_stackchart['Curing_Date'].max() 
            max_date = st.date_input('Select the Date', value=end_data,min_value=start_data, max_value=end_data)
            
            ##################Calling the function
            df_stackchart_trigger,top_df,last_day = Daily_Triggers(df_stackchart, max_date)
      
            
            st.subheader('Overall Rejections - selected day compared to last day')
            second_last_day = df_stackchart_trigger.loc[df_stackchart_trigger['Curing_Date'] ==(max_date -pd.to_timedelta("1day")) ,:].drop(['Curing_Date'], axis=1)
        
        
            col1, col2, col3 = st.columns(3)
            RFPP_overall=last_day.loc[(last_day['Rejection_Reason']=='RFPP') &(last_day['curing_machine']=='Overall')  ,'Rejection_percent']
            Imbalance_overall=last_day.loc[(last_day['Rejection_Reason']=='Imbalance') &(last_day['curing_machine']=='Overall')  ,'Rejection_percent']
            Total_overall=last_day.loc[(last_day['Rejection_Reason']=='Total') &(last_day['curing_machine']=='Overall')  ,'Rejection_percent']
            
            RFPP_secondlast=second_last_day.loc[(second_last_day['Rejection_Reason']=='RFPP') &(second_last_day['curing_machine']=='Overall')  ,'Rejection_percent']
            Imbalance_secondlast=second_last_day.loc[(second_last_day['Rejection_Reason']=='Imbalance') &(second_last_day['curing_machine']=='Overall')  ,'Rejection_percent']
            Total_secondlast=second_last_day.loc[(second_last_day['Rejection_Reason']=='Total') &(second_last_day['curing_machine']=='Overall')  ,'Rejection_percent']
      
            col1.metric('RFPP %',value=round(RFPP_overall,2), delta=round(float(RFPP_overall)-float(RFPP_secondlast),2), delta_color="inverse")
            col2.metric('Imbalance %',value=round(Imbalance_overall,2), delta=round(float(Imbalance_overall)-float(Imbalance_secondlast),2), delta_color="inverse")
            col3.metric('Total %',value= round(Total_overall,2), delta=round(float(Total_overall)-float(Total_secondlast),2), delta_color="inverse")
        
        
            st.subheader('Rejection Trends')
            ###############parameters##############
            cut_off = st.slider('Select the Cut off Percent',min_value=0, max_value=100,value=10) 
            cut_off=cut_off/100        
            lookback_time = st.slider('Select the Look back Time Period',min_value=2, max_value=15,value=7)+3
            ###########################################
            
            
            df_stackchart_trigger,trend_df = Daily_rejection_trends(df_stackchart_trigger, max_date,lookback_time)
           
            trend_df= trend_df.loc[trend_df['Rejection_3DMA'] >= ((1+cut_off) * trend_df['Rejection_avg_3DMA'])  ,:].reset_index(drop=True)
            display_text = 'Cavity Rejections having 3DMA(3 day moving average) rejection of the selected day '+str(round(cut_off *100))+'% greater than\nof last '+ str(lookback_time) +' days average 3DMA is highlighted'
            st.text(display_text)
            st.write(trend_df)
        
            st.subheader('Worst Rejections- selected day')
            st.text('Worst 3 cavity rejections for selected day is highlighted')
            top_df = top_df.loc[top_df['tyres_rejected']>0,:]
            st.write(top_df.loc[top_df['Rejection_Reason']=='Total' ,:])
            st.write(top_df.loc[top_df['Rejection_Reason']=='RFPP' ,:])
            st.write(top_df.loc[top_df['Rejection_Reason']=='Imbalance' ,:])


    st.write("% s seconds to run the code" % round((time.time() - start),3))     
 

image = Image.open('muscle_man2.png')
st.sidebar.image(image)

#############################
#
