import config
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import select, insert
from sqlalchemy import Table, MetaData
from sqlalchemy.exc import SQLAlchemyError

class SqlSession(object):
    def __init__(self):
        self.engine = None
        self._connection = None
        self._data = None
        self.url = config.RDMS + '://' + config.USER + ':' + config.PASSWORD + '@' + config.HOST + '/' + config.DATABASE

    def init(self):
        """
        Initialize Data Source, Connection object
        """
        self.engine = self.create_engine()
        self._connection = self.get_connection()

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
            data = pd.read_sql_query(sql, self._connection)

            return data

        except SQLAlchemyError as e:
            error = str(e.__dict__['orig'])
            return error

    def insert(self, sql: str, data_list: list):
        """
        execute CRUD query
        """
        if self._connection is None:
            raise ConnectionError('Data Source session is not initialized')

        try:
            # cursor = self._connection.cursor()
            # sql_exec = cursor.insert(sql)
            self._connection.execute(sql, data_list)

        except SQLAlchemyError as e:
            error = str(e.__dict__['orig'])
            return error

        finally:
            self.close()