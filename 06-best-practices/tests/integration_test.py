from datetime import datetime
import pandas as pd
import os
import batch

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Functions to test:
# get_input_path(year, month)
# get_output_path(year, month)
# batch.save_data(df_input, output_file, endpoint_url)

def test_save_data_s3():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    year=2023
    month=1
    
    #output_file= batch.get_output_path(year, month)
    output_file= batch.get_input_path(year, month)
        
    endpoint_url = os.getenv('S3_ENDPOINT_URL', None)
    
    batch.save_data(df_input, output_file, endpoint_url)
    