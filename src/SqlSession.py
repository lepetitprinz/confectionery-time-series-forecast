import config
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, MetaData, insert, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


class SqlSession(object):
    def __init__(self):
        # Create Engine Format(Microsoft SQL Server)
        self.url = config.RDMS + '://' + config.USER + ':' + config.PASSWORD + \
                   '@' + config.HOST + ':' + config.PORT + '/' + config.DATABASE

        self.engine = None
        self._connection = None
        self._data = None

    def init(self):
        """
        Initialize Data Source, Connection object
        """
        print("Connect the DB")
        self.engine = self.create_engine()
        self._connection = self.get_connection()
        print("Connected")

    def set_data(self, data_source):
        self._data = data_source

    def create_engine(self):
        return create_engine(self.url)

    def get_connection(self):
        """
        return Data Source Connection
        """
        return self.engine.connect()

    def commit(self):
        """
        commit
        """
        return self._connection.commit()

    def rollback(self):
        """
        rollback
        """
        return self._connection.rollback()

    def close(self):
        """
        close session
        """
        self._connection.close()

    def select(self, sql: str):
        """
        get query
        """
        if self._connection is None:
            raise ConnectionError('Data Source session is not initialized')

        try:
            # result = self._connection.execute(text(sql))
            # columns = [col for col in result.keys()]
            print("Selection process start")
            data = pd.read_sql_query(sql, self._connection)

            return data

        except SQLAlchemyError as e:
            error = str(e.__dict__['orig'])
            return error

    def insert(self, df: pd.DataFrame, tb_name: str):
        table = self.get_table_meta(tb_name=tb_name)
        df.columns = [col.upper() for col in df.columns]
        with self.engine.connect() as conn:
            conn.execute(table.insert(), df.to_dict('records'))

    def update(self, df: pd.DataFrame, tb_name: str):
        table = self.get_table_meta(tb_name=tb_name)
        with self.engine.connect() as conn:
            conn.execute(table.update(), df.to_dict('records'))

    def get_table_meta(self, tb_name: str):
        metadata = MetaData(bind=self.engine)
        metadata.reflect(self.engine, only=[tb_name])
        table = Table(tb_name, metadata, autoload=True, autoload_with=self.engine)

        return table