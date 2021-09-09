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
        SELECT *
        FROM(
            SELECT DIVISION_CD
                 , CUST_CD
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
                         , SOLD_CUST_GRP_CD AS CUST_CD
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
                     WHERE YYMMDD BETWEEN {kwargs['date_from']} AND {kwargs['date_to']}
                    ) SALES
              LEFT OUTER JOIN (
                               SELECT ITEM_CD AS SKU_CD
                                    , ITEM_ATTR01_CD AS BIZ_CD
                                    , ITEM_ATTR02_CD AS LINE_CD
                                    , ITEM_ATTR03_CD AS BRAND_CD
                                    , ITEM_ATTR04_CD AS ITEM_CD
                                 FROM VIEW_I002040
                                WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
                              ) ITEM
                ON SALES.SKU_CD = ITEM.SKU_CD
            ) MST
            WHERE (LINE_CD = 'P111' OR BRAND_CD = 'P304020')    --- EXCEPTION
               """
        return sql

    # SELL-OUT Table
    @staticmethod
    def sql_sell_out(**kwargs):
        sql = f""" 
           SELECT DIVISION_CD
                , CUST_CD
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
                        , SOLD_CUST_GRP_CD AS CUST_CD
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
                                   , ITEM_ATTR01_CD AS BIZ_CD
                                   , ITEM_ATTR02_CD AS LINE_CD
                                   , ITEM_ATTR03_CD AS BRAND_CD
                                   , ITEM_ATTR04_CD AS ITEM_CD
                                FROM VIEW_I002040
                               WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
                             ) ITEM
               ON SALES.SKU_CD = ITEM.SKU_CD
               """
        return sql

    @staticmethod
    def sql_unit_map():
        sql = """
            SELECT BOX.ITEM_CD AS SKU_CD
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
            SELECT LOWER(STAT_CD) AS MODEL
                 , INPUT_POINT AS INPUT_WIDTH
                 , PERIOD AS LABEL_WIDTH
              FROM M4S_I103010
             WHERE USE_YN = 'Y'
               AND DIVISION_CD = '{kwargs['division']}'           
            """
        return sql

    @staticmethod
    def sql_best_hyper_param_grid():
        sql = """
            SELECT STAT_CD
                 , OPTION_CD
                 , OPTION_VAL
              FROM M4S_I103011
            """
        return sql

    @staticmethod
    def sql_err_grp_map():
        sql = """
            SELECT COMM_DTL_CD
                 , ATTR01_VAL
              FROM M4S_I002011
             WHERE COMM_CD = 'ERR_CD'
            """
        return sql

    @staticmethod
    def sql_item_profile():
        sql = """
            SELECT ITEM_CD AS SKU_CD
                 , ITEM_NM AS SKU_NM
                 , BOM_CD
                 , BOM_NM
              FROM M4S_I002043   
        """
        return sql
