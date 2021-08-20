class SqlConfig(object):
    def __init__(self, date_from='', date_to=''):
        self.date_from = date_from
        self.date_to = date_to

    # Item Master Table
    def get_item_master(self):
        sql = f"""
            SELECT PROJECT_CD
                 , ITEM_CD
                 , ITEM_NM
              FROM M4S_I002040
            """
        return sql

    # Customer Master Table
    def get_cust_master(self):
        sql = f"""
            SELECT PROJECT_CD
                 , CUST_CD
                 , CUST_NM
              FROM M4S_I002060
            """
        return sql

    # D
    def get_comm_master(self):
        sql = f"""
            SELECT OPTION_CD
                 , OPTION_VAL
            FROM M4S_I001020
           WHERE 1=1
             AND MDL_CD = 'DF'
        """
        return sql

    # SELL-IN Table
    def get_sell_in(self):
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
             FROM M4S_I002170
            WHERE 1=1
              AND YYMMDD BETWEEN {self.date_from} AND {self.date_to} 
        """
        return sql

    # SELL-OUT Table
    def get_sell_out(self):
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
              AND YYMMDD BETWEEN {self.date_from} AND {self.date_to} 
        """
        return sql
