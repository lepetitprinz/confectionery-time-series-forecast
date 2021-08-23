
class SqlConfig(object):
    # Item Master Table
    @staticmethod
    def get_item_master():
        sql = f"""
            SELECT PROJECT_CD
                 , ITEM_CD
                 , ITEM_NM
              FROM M4S_I002040
            """
        return sql

    # Customer Master Table
    @staticmethod
    def get_cust_master():
        sql = f"""
            SELECT PROJECT_CD
                 , CUST_CD
                 , CUST_NM
              FROM M4S_I002060
            """
        return sql

    # D
    @staticmethod
    def get_comm_master(option: str):
        sql = f"""
            SELECT OPTION_VAL
            FROM M4S_I001020
           WHERE 1=1
             AND MDL_CD = 'DF'
             AND OPTION_CD = '{option}'
        """
        return sql

    # SELL-IN Table
    @staticmethod
    def get_sell_in(date_from: str, date_to: str):
        sql = f""" 
            SELECT PROJECT_CD
                 , DIVISION_CD
                 , SOLD_CUST_GRP_CD AS CUST_GRP
                 , ITEM_CD
                 , YYMMDD
                 , SEQ
                 , FROM_DC_CD
                 , UNIT_PRICE
                 , UNIT_CD
                 , DISCOUNT
                 , WEEK
                 , RST_SALES_QTY
             FROM M4S_I002170
            WHERE 1=1
              AND YYMMDD BETWEEN {date_from} AND {date_to} 
        """
        return sql

    # SELL-OUT Table
    @staticmethod
    def get_sell_out(date_from: str, date_to: str):
        sql = f""" 
            SELECT PROJECT_CD
                 , DIVISION_CD
                 , SOLD_CUST_GRP_CD
                 , ITEM_CD
                 , YYMMDD
                 , FROM_DC_CD
                 , UNIT_PRICE
                 , UNIT_CD
                 , DISCOUNT
                 , WEEK
                 , RST_SALES_QTY
             FROM M4S_I002173
            WHERE 1=1
              AND YYMMDD BETWEEN {date_from} AND {date_to} 
        """
        return sql

    def inst_sell_in(self):
        sql = f""""""

        return sql