from dotenv import load_dotenv

load_dotenv('config/.env')

def query_vector_db(query: str) -> str:
    """This function returns the relevant statistics for a given stock market listed company

    Parameters:
        - company_name: stock name (e.g., 'ABANS FINANCE PLC'), use 'all' to get all stock
        """
    import os

    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions


    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY") ,
                    model_name="text-embedding-3-small"
                )



    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    chroma_client.heartbeat()

    collection = chroma_client.get_or_create_collection(name="test_collection2", embedding_function=openai_ef)

    results = collection.query(
        query_texts=[query], # Chroma will embed this for you
        n_results=2 # how many results to return
    )

    res = ""
    if 'documents' in results and len(results['documents']) > 0:
        
        for doc in results['documents'][0]:
            res += doc + "\n"
    return res


def get_time_series_stock_preds(stock_name: str, forecast_horizon: int=1) -> pd.DataFrame:
    """
    This function returns the forecasted values as a pandas df for a given stock name and forecast horizon.

    Parameters:
        - stock_name: stock name (e.g., 'ABANS FINANCE PLC'), use 'all' to get all stocks
        - forecast_horizon: number of days to forecast
       
    """
    import os
    import pandas as pd

    from statsforecast.core import StatsForecast
    from statsforecast.models import AutoARIMA, Naive
    
    csv_list = [i for i in os.listdir('data') if 'cse-trade' in i]
    df_final = pd.DataFrame()
    for i in csv_list:
        tmp = pd.read_csv('data/'+i)
        tmp['ds'] = i[-14:-4]
        df_final = pd.concat([df_final, tmp], ignore_index=True)

    df_final = df_final.rename(columns={'High (Rs.)':'y', 'Company Name':'unique_id'})
    df_final = df_final[['ds', 'unique_id', 'y']]
    df_final['ds'] = pd.to_datetime(df_final['ds'])

    if stock_name == 'all':
        pass
    else:
        df_final = df_final.query('unique_id == @stock_name')
        
    fcst = StatsForecast(models=[AutoARIMA(season_length=4)],
                     freq='D', n_jobs=-1)
    y_hat_df = fcst.forecast(df=df_final, h=forecast_horizon)
    y_hat_df = y_hat_df.rename(columns={'AutoARIMA':'forecasted_value'})
    return y_hat_df.to_markdown()