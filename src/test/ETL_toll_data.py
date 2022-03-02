from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

# Task 1.1 
# Define DAG arguments
default_args={
    'owner': 'me',
    'start_date': datetime.now(),
    'email': 'abc@gmail.com',
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Task 1.2
# Define the DAG
dag = DAG(
    'ETL_toll_data',
    description='Apache Airflow Final Assignment',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)

# Task 1.3
# Create a task to unzip data
task1 = BashOperator(
    task_id='unzip_data',
    bash_command='tar -xvzf /home/project/airflow/tolldata.tgz -C /home/project/airflow/dags/finalassignment/staging',
    dag=dag
)

# Task1.4
# Create a task to extract data from csv file
def extract_data_from_csv():
    data = pd.read_csv(
        '/home/project/airflow/dags/finalassignment/staging/vehicle-data.csv', 
        header=None
    )
    data.columns = [
        'Rowid', 'Timestamp', 'Anonymized Vehicle number', 'Vehicle type', 'Number of axles', 'Vehicle code']
    data_filtered = data[['Rowid', 'Timestamp', 'Anonymized Vehicle number', 'Vehicle type']]
    data_filtered.to_csv('csv_data.csv')
    
task2 = PythonOperator(
    task_id='extract_data_from_csv',
    provide_context=True,
    python_callable=extract_data_from_csv,
    dag=dag
)

# Task1.5
# Create a task to extract data from tsv file
def extract_data_from_tsv():
    data = pd.read_csv(
        '/home/project/airflow/dags/finalassignment/staging/tollplaza-data.tsv', 
        header=None,
        sep='\t'
    )
    data.columns = [
        'Rowid', 'Timestamp', 'Anonymized Vehicle number', 'Vehicle type', 'Number of axles', 
        'Tollplaza id', 'Tollplaza code']
    data_filtered = data[['Number of axles', 'Tollplaza id', 'Tollplaza code']]
    data_filtered.to_csv('tsv_data.csv')
    
task3 = PythonOperator(
    task_id='extract_data_from_tsv',
    provide_context=True,
    python_callable=extract_data_from_tsv,
    dag=dag
)

# Task1.6
# Create a task to extract data from fixed width
def extract_data_from_fixed_width():
    data = pd.read_fwf(
        '/home/project/airflow/dags/finalassignment/staging/payment-data.txt', 
        header=None
    )
    data.columns = [str(col) for col in data.columns]
    data['Timestamp'] = data['1'] + ' ' + data['2'] + ' ' + data['3'].astype(str) + ' ' + data['4'] + ' ' + data['5'].astype(str)
    data = data.drop(columns=['1', '2', '3', '4', '5'])
    data = data.rename(columns={
        '0': 'Rowid', '6': 'Anonymized Vehicle number', '7': 'Tollplaza id',
        '8': 'Tollplaza code', '9': 'Type of Payment code', '10': 'Vehicle Code'})
    data.head()
    data_filtered = data[['Type of Payment code', 'Vehicle Code']]
    data_filtered.to_csv('fixed_width_data.csv')
    
task4 = PythonOperator(
    task_id='extract_data_from_fixed_width',
    provide_context=True,
    python_callable=extract_data_from_fixed_width,
    dag=dag
)

# Task1.7
# Create a task to consolidate data extracted from previous tasks
task5 = BashOperator(
    task_id='consolidate_data',
    bash_command='paste -s csv_data.csv tsv_data.csv fixed_width_data.csv > merged_data.csv',
    dag=dag
)

# Task1.8
# Transform and load the data
def transform_data():
    data = pd.read_csv('/home/project/airflow/dags/finalassignment/staging/merged_data.csv')
    data['Vehicle type'] = data['Vehicle type'].str.upper()
    data.to_csv('transformed_data.csv')
    
task6 = PythonOperator(
    task_id='transform_data',
    provide_context=True,
    python_callable=transform_data,
    dag=dag
)

# Task1.9
# Define the task pipeline
task1 >> task2 >> task3 >> task4 >> task5 >> task6