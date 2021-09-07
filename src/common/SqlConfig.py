
class SqlConfig(object):
    # Item Master Table
    @staticmethod
    def sql_item_master():
        sql = f"""
            SELECT PROJECT_CD
                 , ITEM_CD
                 , ITEM_NM
              FROM M4S_I002040
            """
        return sql

    # Customer Master Table
    @staticmethod
    def sql_cust_master():
        sql = f"""
            SELECT PROJECT_CD
                 , CUST_CD
                 , CUST_NM
              FROM M4S_I002060
            """
        return sql

    # D
    @staticmethod
    def sql_comm_master():
        sql = f"""
            SELECT OPTION_CD
                 , OPTION_VAL
            FROM M4S_I001020
           WHERE MDL_CD = 'DF'
            """
        return sql

    # SELL-IN Table
    @staticmethod
    def sql_sell_in(**kwargs):
        sql = f""" 
            SELECT DIVISION_CD
                 , SOLD_CUST_GRP_CD
                 , BIZ_CD
                 , LINE_CD
                 , BRAND_CD
                 , ITEM_CD
                 , SALES.SKU_CD
                 , YYMMDD
                 , SEQ
                 , FROM_DC_CD
                 , UNIT_PRICE
                 , UNIT_CD
                 , DISCOUNT
                 , WEEK
                 , QTY
                 , CREATE_DATE
              FROM (
                    SELECT PROJECT_CD
                         , DIVISION_CD
                         , SOLD_CUST_GRP_CD
                         , ITEM_CD AS SKU_CD
                         , YYMMDD
                         , SEQ
                         , FROM_DC_CD
                         , UNIT_PRICE
                         , UNIT_CD
                         , DISCOUNT
                         , WEEK
                         , RST_SALES_QTY AS QTY
                         , CREATE_DATE
                      FROM M4S_I002170
                     WHERE FROM_DC_CD NOT LIKE '%공통%'  -- Exception
                       AND YYMMDD BETWEEN {kwargs['date_from']} AND {kwargs['date_to']}
                    ) SALES
              LEFT OUTER JOIN (
                               SELECT ITEM_CD AS SKU_CD
                                    , ITEM_GUBUN01_CD AS BIZ_CD
                                    , ITEM_GUBUN02_CD AS LINE_CD
                                    , ITEM_GUBUN03_CD AS BRAND_CD
                                    , ITEM_GUBUN04_CD AS ITEM_CD
                                 FROM VIEW_I002040
                                WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
                              ) ITEM
                ON SALES.SKU_CD = ITEM.SKU_CD
               """
        return sql

    # SELL-OUT Table
    @staticmethod
    def sql_sell_out(**kwargs):
        sql = f""" 
           SELECT DIVISION_CD
                , SOLD_CUST_GRP_CD
                , BIZ_CD
                , LINE_CD
                , BRAND_CD
                , ITEM_CD
                , SALES.SKU_CD
                , YYMMDD
                , SEQ
                , DISCOUNT
                , WEEK
                , QTY
                , CREATE_DATE
             FROM (
                   SELECT PROJECT_CD
                        , DIVISION_CD
                        , SOLD_CUST_GRP_CD
                        , ITEM_CD AS SKU_CD
                        , YYMMDD
                        , SEQ
                        , DISCOUNT
                        , WEEK
                        , RST_SALES_QTY AS QTY
                        , CREATE_DATE
                     FROM M4S_I002173
                    WHERE YYMMDD BETWEEN {kwargs['date_from']} AND {kwargs['date_to']}
                   ) SALES
             LEFT OUTER JOIN (
                              SELECT ITEM_CD AS SKU_CD
                                   , ITEM_GUBUN01_CD AS BIZ_CD
                                   , ITEM_GUBUN02_CD AS LINE_CD
                                   , ITEM_GUBUN03_CD AS BRAND_CD
                                   , ITEM_GUBUN04_CD AS ITEM_CD
                                FROM VIEW_I002040
                               WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
                             ) ITEM
               ON SALES.SKU_CD = ITEM.SKU_CD
               """
        return sql

    @staticmethod
    def sql_unit_map():
        sql = """
            SELECT BOX.ITEM_CD
                 , CONVERT(INT, BOX.FAC_PRICE / BOL.FAC_PRICE) AS BOX_BOL
                 , CONVERT(INT, BOX.FAC_PRICE / EA.FAC_PRICE) AS BOX_EA
              FROM (
                    SELECT PROJECT_CD
                         , ITEM_CD
                         , PRICE_START_YYMMDD
                         , FAC_PRICE
                    FROM M4S_I002041
                   WHERE PRICE_QTY_UNIT_CD = 'BOX'
                     AND FAC_PRICE <> 0
                    ) BOX
              LEFT OUTER JOIN (
                               SELECT PROJECT_CD
                                    , ITEM_CD
                                    , PRICE_START_YYMMDD
                                    , FAC_PRICE
                                 FROM M4S_I002041
                                WHERE PRICE_QTY_UNIT_CD = 'BOL'
                                  AND FAC_PRICE <> 0
                              ) BOL
                ON BOX.PROJECT_CD = BOL.PROJECT_CD
               AND BOX.ITEM_CD = BOL.ITEM_CD
               AND BOX.PRICE_START_YYMMDD = BOL.PRICE_START_YYMMDD
              LEFT OUTER JOIN (
                               SELECT PROJECT_CD
                                    , ITEM_CD
                                    , PRICE_START_YYMMDD
                                    , FAC_PRICE
                                 FROM M4S_I002041
                                WHERE PRICE_QTY_UNIT_CD = 'EA'
                                  AND FAC_PRICE <> 0
                               ) EA
                ON BOX.PROJECT_CD = EA.PROJECT_CD
               AND BOX.ITEM_CD = EA.ITEM_CD
               AND BOX.PRICE_START_YYMMDD = EA.PRICE_START_YYMMDD
            """
        return sql

    @staticmethod
    def sql_algorithm(**kwargs):
        sql = f"""
            SELECT LOWER(STAT) AS MODEL
                 , INPUT_POINT AS INPUT_WIDTH
                 , PERIOD AS LABEL_WIDTH
              FROM M4S_I103010
             WHERE USE_YN = 'Y'
               AND DIVISION = '{kwargs['division']}'           
            """
        return sql

    @staticmethod
    def sql_best_hyper_param_grid():
        sql = """
            SELECT STAT
                 , OPTION_CD
                 , OPTION_VAL
              FROM M4S_I103011
            """
        return sql
