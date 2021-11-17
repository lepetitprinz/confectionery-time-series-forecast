import common.config as config

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, MetaData, insert
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError


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
        try:
            self._connection.close()
            print("DB Session is closed")

        except SQLAlchemyError as e:
            error = str(e.__dict__['orig'])
            print(f"Error: {error}")

    def select(self, sql: str):
        """
        get query
        """
        if self._connection is None:
            raise ConnectionError('Session is not initialized')

        try:
            data = pd.read_sql_query(sql, self._connection)
            return data

        except SQLAlchemyError as e:
            error = str(e)
            return error

    def insert(self, df: pd.DataFrame, tb_name: str):
        # Get meta information
        table = self.get_table_meta(tb_name=tb_name)
        df.columns = [col.upper() for col in df.columns]

        with self.engine.connect() as conn:
            conn.execute(table.insert(), df.to_dict('records'))
            print(f"Saving {tb_name} table is finished.")

    def delete(self, sql: str):
        statement = text(sql)
        with self.engine.connect() as conn:
            conn.execute(statement)

    def upsert(self, df: pd.DataFrame, tb_name: str):
        table = self.get_table_meta(tb_name=tb_name)
        stmt = insert(table).values(df.to_dict('records'))
        stmt = stmt.on_conflict_do_update(
            constraint='post_key',
            set_={}
            )
        with self.engine.connect() as conn:
            conn.execute(stmt)

    def get_table_meta(self, tb_name: str):
        metadata = MetaData(bind=self.engine)
        metadata.reflect(self.engine, only=[tb_name])
        table = Table(tb_name, metadata, autoload=True, autoload_with=self.engine)

        return table

    # @compiles(Insert)
    # def compile_upsert(self, insert_stmt, compiler, **kwargs):
    #     pk = insert_stmt.table.primary_key
    #     insert = compiler.visit_insert(insert_stmt, **kwargs)
    #     ondup = f'ON CONFLICT ({",".join(c.name for c in pk)}) DO UPDATE SET'
    #     updates = ', '.join(f"{c.name}=EXCLUDED.{c.name}" for c in insert_stmt.table.columns)
    #     upsert = ' '.join((insert, ondup, updates))
    #
    #     return upsert
