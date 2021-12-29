
class SqlConfig(object):
    ######################################
    # Master Data
    ######################################
    # Data Version Management
    @staticmethod
    def sql_data_version():
        sql = f"""
            SELECT DATA_VRSN_CD
                 , EXEC_DATE
              FROM M4S_I110420
            """
        return sql

    # common master
    @staticmethod
    def sql_comm_master():
        sql = f"""
            SELECT OPTION_CD
                 , OPTION_VAL
              FROM M4S_I001020
             WHERE MDL_CD = 'DF'
            """
        return sql

    # Calender master
    @staticmethod
    def sql_calendar():
        sql = """
            SELECT YYMMDD
                 , YY
                 , YYMM
                 , WEEK
                 , START_WEEK_DAY
                 , END_WEEK_DAY
              FROM M4S_I002030
        """
        return sql

    # Item Master
    @staticmethod
    def sql_item_view():
        sql = """
            SELECT ITEM_ATTR01_CD AS BIZ_CD
                 , ITEM_ATTR01_NM AS BIZ_NM
                 , ITEM_ATTR02_CD AS LINE_CD
                 , ITEM_ATTR02_NM AS LINE_NM
                 , ITEM_ATTR03_CD AS BRAND_CD
                 , ITEM_ATTR03_NM AS BRAND_NM
                 , ITEM_ATTR04_CD AS ITEM_CD
                 , ITEM_ATTR04_NM AS ITEM_NM
                 , ITEM_CD AS SKU_CD
                 , ITEM_NM AS SKU_NM
              FROM VIEW_I002040
             WHERE ITEM_TYPE_CD IN ('FERT', 'HAWA')
               AND USE_YN = 'Y'
             GROUP BY ITEM_ATTR01_CD
                    , ITEM_ATTR01_NM
                    , ITEM_ATTR02_CD
                    , ITEM_ATTR02_NM
                    , ITEM_ATTR03_CD
                    , ITEM_ATTR03_NM
                    , ITEM_ATTR04_CD
                    , ITEM_ATTR04_NM
                    , ITEM_CD
                    , ITEM_NM
        """
        return sql

    # SP1 Master
    @staticmethod
    def sql_cust_grp_info():
        sql = """
              SELECT LINK_SALES_MGMT_CD AS CUST_GRP_CD
                   , LINK_SALES_MGMT_NM AS CUST_GRP_NM
                FROM M4S_I204020
               WHERE PROJECT_CD = 'ENT001'
                 AND SALES_MGMT_TYPE_CD = 'SP1'
                 AND SALES_MGMT_VRSN_ID = (SELECT SALES_MGMT_VRSN_ID FROM M4S_I204010 WHERE USE_YN = 'Y')
                 AND USE_YN = 'Y'
        """
        return sql

    @staticmethod
    def sql_bom_mst():
        sql = """
            SELECT ITEM_CD AS SKU_CD
                 , ITEM_NM AS SKU_NM
                 , BOM_CD
                 , BOM_NM
              FROM M4S_I002043   
        """
        return sql

    @staticmethod
    def sql_algorithm(**kwargs):
        sql = f"""
            SELECT LOWER(STAT_CD) AS MODEL
                 , INPUT_POINT AS INPUT_WIDTH
                 , PERIOD AS LABEL_WIDTH
                 , VARIATE
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

    ######################################
    # Sales dataset
    ######################################
    # SELL-IN Table
    @staticmethod
    def sql_sell_in(**kwargs):
        sql = f"""
            SELECT DIVISION_CD
                 , CUST_GRP_CD
                 , ITEM_ATTR01_CD AS BIZ_CD
                 , ITEM_ATTR02_CD AS LINE_CD
                 , ITEM_ATTR03_CD AS BRAND_CD
                 , ITEM_ATTR04_CD AS ITEM_CD
                 , ITEM_CD AS SKU_CD
                 , YYMMDD
                 , SEQ
                 , FROM_DC_CD
                 , UNIT_PRICE
                 , UNIT_CD
                 , DISCOUNT
                 , WEEK
                 , QTY
                 , CREATE_DATE
             FROM M4S_I002176
            WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
        """
        return sql

    @staticmethod
    def sql_sell_in_week_grp(**kwargs):
        sql = f""" 
            SELECT DIVISION_CD
                 , CUST_GRP_CD
                 , SKU_CD
                 , YY
                 , WEEK
                 , SUM(RST_SALES_QTY) AS SALES
              FROM (
                    SELECT DIVISION_CD
                         , CUST.CUST_GRP_CD
                         , SKU_CD
                         , YY
                         , WEEK
                         , RST_SALES_QTY
                      FROM (
                            SELECT DIVISION_CD
                                 , SOLD_CUST_GRP_CD AS CUST_CD
                                 , ITEM_CD          as SKU_CD
                                 , YY
                                 , WEEK
                                 , RST_SALES_QTY
                              FROM M4S_I002170
                             WHERE 1 = 1
                               AND YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                               AND RST_SALES_QTY <> 0 -- Remove 0 quantity
                           ) SALES
                     INNER JOIN (
                                 SELECT CUST_CD
                                      , CUST_GRP_CD
                                   FROM M4S_I002060
                                ) CUST
                        ON SALES.CUST_CD = CUST.CUST_CD
                   ) SALES--      
             GROUP BY DIVISION_CD
                    , CUST_GRP_CD
                    , SKU_CD
                    , YY
                    , WEEK
             """
        return sql

    # SELL-OUT
    @staticmethod
    def sql_sell_out_week(**kwargs):
        sql = f""" 
           SELECT DIVISION_CD
                , CUST_GRP_CD
                , ITEM_ATTR01_CD AS BIZ_CD
                , ITEM_ATTR02_CD AS LINE_CD
                , ITEM_ATTR03_CD AS BRAND_CD
                , ITEM_ATTR04_CD AS ITEM_CD
                , ITEM_CD AS SKU_CD 
                , YYMMDD
                , SEQ
                , UNIT_PRICE
                , UNIT_CD
                , DISCOUNT
                , WEEK
                , QTY
                , CREATE_DATE
             FROM M4S_I002177
            WHERE QTY <> 0
              AND CUST_GRP_CD <> '1173'
              AND YYMMDD BETWEEN {kwargs['from']} AND {kwargs['to']}
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
    def sql_err_grp_map():
        sql = """
            SELECT COMM_DTL_CD
                 , ATTR01_VAL
              FROM M4S_I002011
             WHERE COMM_CD = 'ERR_CD'
            """
        return sql

    @staticmethod
    def sql_exg_data(partial_yn: str):
        sql = f"""
            SELECT IDX_CD
                 , IDX_DTL_CD
                 , YYMM
                 , REF_VAL
              FROM M4S_O110710
             WHERE IDX_CD IN (
                              SELECT IDX_CD
                                FROM M4S_O110700
                               WHERE USE_YN = 'Y'
                                 AND EXG_ID IN (
                                                SELECT EXG_ID
                                                  FROM M4S_O110701
                                                 WHERE USE_YN = 'Y'
                                                 AND PARTIAL_YN = '{partial_yn}'
                                               )
                             )
        """
        return sql

    @staticmethod
    def sql_pred_item(**kwargs):
        sql = f"""
            SELECT DATA_VRSN_CD
                 , DIVISION_CD
                 , CUST_GRP_CD
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , ITEM_CD
                 , WEEK
                 , YYMMDD
                 , SUM(RESULT_SALES) AS QTY
              FROM (
                    SELECT DATA_VRSN_CD
                         , DIVISION_CD
                         , WEEK
                         , YYMMDD
                         , RESULT_SALES
                         , CUST_GRP_CD
                         , ITEM_ATTR01_CD
                         , ITEM_ATTR02_CD
                         , ITEM_ATTR03_CD
                         , ITEM_ATTR04_CD
                         , ITEM_CD
                      FROM M4S_O110600
                     WHERE DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
                       AND DIVISION_CD = '{kwargs['division_cd']}'
                       AND LEFT(FKEY, 5) = '{kwargs['fkey']}'
                       AND ITEM_CD ='{kwargs['item_cd']}'
                       AND CUST_GRP_CD ='{kwargs['cust_grp_cd']}'
                  ) PRED
             GROUP BY DATA_VRSN_CD
                    , DIVISION_CD
                    , YYMMDD
                    , WEEK
                    , CUST_GRP_CD
                    , ITEM_ATTR01_CD
                    , ITEM_ATTR02_CD
                    , ITEM_ATTR03_CD
                    , ITEM_ATTR04_CD
                    , ITEM_CD
                """
        return sql

    @staticmethod
    def sql_sales_item(**kwargs):
        sql = f"""
            SELECT * 
              FROM (
                    SELECT CAL.YYMMDD
                         , RST_SALES_QTY AS QTY_LAG
                      FROM (
                            SELECT *
                              FROM M4S_I002175
                             WHERE DIVISION_CD = '{kwargs['division_cd']}'
                               AND CUST_GRP_CD = '{kwargs['cust_grp_cd']}'
                               AND ITEM_CD = '{kwargs['item_cd']}'
                           ) SALES
                      LEFT OUTER JOIN (
                                       SELECT START_WEEK_DAY AS YYMMDD
                                            , YY
                                            , WEEK
                                         FROM M4S_I002030
                                        GROUP BY START_WEEK_DAY
                                               , YY
                                               , WEEK
                                      ) CAL
                        ON SALES.YYMMDD = CAL.YY
                       AND SALES.WEEK = CAL.WEEK
                   ) RSLT
             WHERE YYMMDD BETWEEN '{kwargs['from_date']}' AND '{kwargs['to_date']}'
              """

        return sql

    @staticmethod
    def sql_data_level():
        sql = """
            select S_COL02
                 , S_COL03
                 , S_COL04
                 , S_COL05
                 , S_COL06
            from M4S_I103030
        """
        return sql

    @staticmethod
    def sql_what_if_exec_info():
        sql = """
            SELECT DATA_VRSN_CD
                 , DIVISION_CD
                 , WI_VRSN_ID
                 , WI_VRSN_SEQ
                 , SALES_MGMT_CD
                 , ITEM_CD
                 , CREATE_USER_CD
              FROM M4S_I110520
             WHERE EXEC_YN = 'P'
        """
        return sql

    @staticmethod
    def sql_what_if_exec_list():
        sql = """
            SELECT DTL.DATA_VRSN_CD
                 , DTL.DIVISION_CD
                 , DTL.WI_VRSN_ID
                 , DTL.WI_VRSN_SEQ
                 , DTL.SALES_MGMT_CD
                 , DTL.ITEM_CD
                 , DTL.YY
                 , DTL.YYMM
                 , DTL.WEEK
                 , DISCOUNT / 100 AS DISCOUNT
                 , DTL.CREATE_USER_CD
              FROM M4S_I110521 DTL
             INNER JOIN (
                         SELECT *
                           FROM M4S_I110520
                          WHERE EXEC_YN = 'P'
                        ) MST
                ON DTL.DATA_VRSN_CD = MST.DATA_VRSN_CD
               AND DTL.DIVISION_CD = MST.DIVISION_CD
               AND DTL.WI_VRSN_ID = MST.WI_VRSN_ID
               AND DTL.WI_VRSN_SEQ = MST.WI_VRSN_SEQ
               AND DTL.SALES_MGMT_CD = MST.SALES_MGMT_CD
               AND DTL.ITEM_CD = MST.ITEM_CD
               AND DTL.CREATE_USER_CD = MST.CREATE_USER_CD
        """
        return sql

    @staticmethod
    def sql_old_item_list():
        sql = """
            SELECT ITEM_CD 
              FROM M4S_I002040
             WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
               AND NEW_ITEM_YN = 'N'
               AND ITEM_NM NOT LIKE '%삭제%'
        """
        return sql

    ######################################
    # Update Query
    ######################################
    @staticmethod
    def update_data_version(**kwargs):
        sql = f"""
            UPDATE M4S_I110420
               SET USE_YN = 'N'
             WHERE DATA_VRSN_CD <> '{kwargs['data_vrsn_cd']}'
        """
        return sql

    @staticmethod
    def update_what_if_exec_info(**kwargs):
        sql = f"""
            UPDATE M4S_I110520
               SET EXEC_YN = 'Y'
             WHERE DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND WI_VRSN_ID = '{kwargs['wi_vrsn_id']}'
               AND WI_VRSN_SEQ = '{kwargs['wi_vrsn_seq']}'
               AND SALES_MGMT_CD = '{kwargs['sales_mgmt_cd']}'
               AND ITEM_CD = '{kwargs['item_cd']}'
               AND CREATE_USER_CD = '{kwargs['create_user_cd']}'
        """
        return sql

    ######################################
    # Delete Query
    ######################################
    @staticmethod
    def del_sales_err(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_I002174
             WHERE DATA_VRSN_CD =  '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND ERR_CD = '{kwargs['err_cd']}'
        """
        return sql

    @staticmethod
    def del_openapi(**kwargs):
        sql = f"""
            DELETE 
              FROM M4S_O110710
             WHERE PROJECT_CD = 'ENT001'
               AND IDX_CD = '{kwargs['idx_cd']}'
               AND IDX_DTL_CD = '{kwargs['idx_dtl_cd']}'
               AND YYMM BETWEEN '{kwargs['api_start_day']}' AND '{kwargs['api_end_day']}'
        """
        return sql

    @staticmethod
    def del_score(**kwargs):
        sql = f"""
            DELETE 
              FROM {kwargs['table_nm']}
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND LEFT(FKEY, 5) = '{kwargs['fkey']}'
        """
        return sql

    @staticmethod
    def del_profile(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_O110300
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
        """
        return sql

    @staticmethod
    def del_pred_all(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_I110400
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND LEFT(FKEY, 5) = '{kwargs['fkey']}'
        """
        return sql

    @staticmethod
    def del_pred_best(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_O110600
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND LEFT(FKEY, 5) = '{kwargs['fkey']}'
        """
        return sql

    @staticmethod
    def del_hyper_params(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_I103011
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND STAT_CD = '{kwargs['stat_cd']}'
               AND OPTION_CD = '{kwargs['option_cd']}'
        """
        return sql

    @staticmethod
    def del_decomposition(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_O110500
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               and HRCHY_LVL_CD = '{kwargs['hrchy_lvl_cd']}'
        """
        return sql

    @staticmethod
    def del_compare_result(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_O110620
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND TEST_VRSN_CD = '{kwargs['test_vrsn_cd']}'
        """
        return sql

    @staticmethod
    def del_sim_result(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_I110521
             WHERE PROJECT_CD = '{kwargs['project_cd']}'
               AND DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND WI_VRSN_ID = '{kwargs['wi_vrsn_id']}'
               AND WI_VRSN_SEQ = '{kwargs['wi_vrsn_seq']}'
               AND SALES_MGMT_CD = '{kwargs['sales_mgmt_cd']}'
               AND ITEM_CD = '{kwargs['item_cd']}'
               AND YY = '{kwargs['yy']}'
               AND WEEK = '{kwargs['week']}'
               AND CREATE_USER_CD = '{kwargs['create_user_cd']}'
        """
        return sql

    @staticmethod
    def del_pred_recent():
        sql = f"""
            DELETE
              FROM M4S_O111600
        """
        return sql

    @staticmethod
    def sql_sell_in_unit(**kwargs):
        sql = f""" 
              SELECT DIVISION_CD
                   , CUST_GRP_CD
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
                           , SOLD_CUST_GRP_CD AS CUST_GRP_CD
                           , ITEM_CD AS SKU_CD
                           , YYMMDD
                           , SEQ
                           , FROM_DC_CD
                           , UNIT_PRICE
                           , RTRIM(UNIT_CD) AS UNIT_CD
                           , DISCOUNT
                           , WEEK
                           , RST_SALES_QTY AS QTY
                           , CREATE_DATE
                        FROM M4S_I002170_TEST
                       WHERE YYMMDD BETWEEN {kwargs['from']} AND {kwargs['to']}
                       AND RST_SALES_QTY > 0 
                       AND SOLD_CUST_GRP_CD = '{kwargs['cust_grp_cd']}',
                       AND ITEM_CD = '{kwargs['item_cd']}'
                       --and SOLD_CUST_GRP_CD = '1033' -- exception
                      ) SALES
               INNER JOIN (
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
    def sql_sell_out_week_grp(**kwargs):
        sql = f"""
            SELECT DIVISION_CD
                 , CUST_GRP_CD
                 , ITEM_CD AS SKU_CD
                 , YY
                 , WEEK
                 , ISNULL(SUM(QTY), 0) AS SALES
              FROM (
                    SELECT DIVISION_CD
                         , CUST_GRP_CD
                         , ITEM_CD
                         , SUBSTRING(YYMMDD, 1, 4) AS YY
                         , WEEK
                         , QTY
                      FROM M4S_I002177
                     WHERE CUST_GRP_CD <> '1173'
                       AND YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                   ) SALES
             GROUP BY DIVISION_CD
                    , CUST_GRP_CD
                    , ITEM_CD
                    , YY
                    , WEEK
        """
        return sql

    @staticmethod
    def sql_weather_avg(**kwargs):
        sql = f"""
            SELECT PROJECT_CD
                 , IDX_CD
                 , '999' AS IDX_DTL_CD
                 , '전국' AS IDX_DTL_NM
                 , YYMM
                 , ROUND(AVG(REF_VAL), 2) AS REF_VAL
                 , 'SYSTEM' AS CREATE_USER_CD
              FROM (
                    SELECT PROJECT_CD
                         , IDX_CD
                         , YYMM
                         , REF_VAL
                      FROM M4S_O110710
                     WHERE 1 = 1
                       AND IDX_DTL_CD IN (108, 112, 133, 143, 152, 156, 159)    -- region
                       AND IDX_CD IN ('TEMP_MIN', 'TEMP_AVG', 'TEMP_MAX', 'RAIN_SUM', 'GSR_SUM', 'RHM_SUM')
                       AND YYMM BETWEEN '{kwargs['api_start_day']}' AND '{kwargs['api_end_day']}'
                   ) WEATHER
             GROUP BY PROJECT_CD
                    , IDX_CD
                    , YYMM
        """
        return sql

    # Sell-Out Monthly Group
    @staticmethod
    def sql_sell_out_month_grp_test(**kwargs):
        pass

    ###################
    # Temp & Test Query
    ###################
    # @staticmethod
    # def sql_sell_in_test(**kwargs):
    #     sql = f"""
    #         SELECT DIVISION_CD
    #              , CUST_GRP_CD
    #              , BIZ_CD
    #              , LINE_CD
    #              , BRAND_CD
    #              , ITEM_CD
    #              , SALES.SKU_CD
    #              , YYMMDD
    #              , SEQ
    #              , FROM_DC_CD
    #              , UNIT_PRICE
    #              , UNIT_CD
    #              , DISCOUNT
    #              , WEEK
    #              , QTY
    #              , CREATE_DATE
    #           FROM (
    #                 SELECT DIVISION_CD
    #                      , SOLD_CUST_GRP_CD AS CUST_GRP_CD
    #                      , ITEM_CD AS SKU_CD
    #                      , YYMMDD
    #                      , SEQ
    #                      , FROM_DC_CD
    #                      , UNIT_PRICE
    #                      , RTRIM(UNIT_CD) AS UNIT_CD
    #                      , DISCOUNT
    #                      , WEEK
    #                      , RST_SALES_QTY AS QTY
    #                      , CREATE_DATE
    #                   FROM M4S_I002170_TEST
    #                  WHERE YYMMDD BETWEEN {kwargs['from']} AND {kwargs['to']}
    #                 ) SALES
    #          INNER JOIN (
    #                      SELECT ITEM_CD AS SKU_CD
    #                           , ITEM_ATTR01_CD AS BIZ_CD
    #                           , ITEM_ATTR02_CD AS LINE_CD
    #                           , ITEM_ATTR03_CD AS BRAND_CD
    #                           , ITEM_ATTR04_CD AS ITEM_CD
    #                        FROM VIEW_I002040
    #                       WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
    #                     ) ITEM
    #             ON SALES.SKU_CD = ITEM.SKU_CD
    #            """
    #
    #     return sql

    # @staticmethod
    # def sql_sell_in_week_grp_test_inqty(**kwargs):
    #     sql = f"""
    #         SELECT DIVISION_CD
    #              , SOLD_CUST_GRP_CD AS CUST_GRP_CD
    #              , SKU_CD
    #              , YY
    #              , WEEK
    #              , SUM(RST_SALES_QTY) AS SALES
    #           FROM (
    #                 SELECT DIVISION_CD
    #                      , SOLD_CUST_GRP_CD
    #                      , SKU_CD
    #                      , YY
    #                      , WEEK
    #                      , CASE WHEN UNIT_CD = 'BOX' THEN RST_SALES_QTY
    #                             WHEN UNIT_CD = 'EA' THEN ROUND(RST_SALES_QTY / BOX_EA, 2)
    #                             WHEN UNIT_CD = 'BOL' THEN ROUND(RST_SALES_QTY / BOX_BOL, 2)
    #                             ELSE 0
    #                              END AS RST_SALES_QTY
    #                   FROM (
    #                         SELECT DIVISION_CD
    #                              , SOLD_CUST_GRP_CD
    #                              , SALES.SKU_CD
    #                              , YY
    #                              , WEEK
    #                              , UNIT_CD
    #                              , BOX_BOL
    #                              , BOX_EA
    #                              , RST_SALES_QTY
    #                           FROM (
    #                                 SELECT DIVISION_CD
    #                                      , SOLD_CUST_GRP_CD
    #                                      , ITEM_CD as SKU_CD
    #                                      , YY
    #                                      , WEEK
    #                                      , UNIT_CD
    #                                      , RST_SALES_QTY
    #                                   FROM M4S_I002170_INQTY
    #                                  WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
    #                                ) SALES
    #           LEFT OUTER JOIN (
    #                            SELECT BOX.ITEM_CD                                 AS SKU_CD
    #                                 , CONVERT(INT, BOX.FAC_PRICE / BOL.FAC_PRICE) AS BOX_BOL
    #                                 , CONVERT(INT, BOX.FAC_PRICE / EA.FAC_PRICE)  AS BOX_EA
    #                              FROM (
    #                                    SELECT PROJECT_CD
    #                                         , ITEM_CD
    #                                         , PRICE_START_YYMMDD
    #                                         , FAC_PRICE
    #                                      FROM M4S_I002041
    #                                     WHERE PRICE_QTY_UNIT_CD = 'BOX'
    #                                       AND FAC_PRICE <> 0
    #                                   ) BOX
    #                              LEFT OUTER JOIN (
    #                                               SELECT PROJECT_CD
    #                                                    , ITEM_CD
    #                                                    , PRICE_START_YYMMDD
    #                                                    , FAC_PRICE
    #                                                 FROM M4S_I002041
    #                                                WHERE PRICE_QTY_UNIT_CD = 'BOL'
    #                                                  AND FAC_PRICE <> 0
    #                                              ) BOL
    #                                ON BOX.PROJECT_CD = BOL.PROJECT_CD
    #                               AND BOX.ITEM_CD = BOL.ITEM_CD
    #                               AND BOX.PRICE_START_YYMMDD = BOL.PRICE_START_YYMMDD
    #                              LEFT OUTER JOIN (
    #                                               SELECT PROJECT_CD
    #                                                    , ITEM_CD
    #                                                    , PRICE_START_YYMMDD
    #                                                    , FAC_PRICE
    #                                                 FROM M4S_I002041
    #                                                WHERE PRICE_QTY_UNIT_CD = 'EA'
    #                                                  AND FAC_PRICE <> 0
    #                                              ) EA
    #                               ON BOX.PROJECT_CD = EA.PROJECT_CD
    #                              AND BOX.ITEM_CD = EA.ITEM_CD
    #                              AND BOX.PRICE_START_YYMMDD = EA.PRICE_START_YYMMDD
    #                           ) UNIT
    #             ON SALES.SKU_CD = UNIT.SKU_CD
    #                      ) SALES
    #              ) SALES
    #          GROUP BY DIVISION_CD
    #                 , SOLD_CUST_GRP_CD
    #                 , SKU_CD
    #                 , YY
    #                 , WEEK
    #     """
    #     return sql

    # @staticmethod
    # def sql_sell_in_week_grp_test(**kwargs):
    #     sql = f"""
    #            SELECT DIVISION_CD
    #                 , SOLD_CUST_GRP_CD AS CUST_GRP_CD
    #                 , SKU_CD
    #                 , YY
    #                 , WEEK
    #                 , SUM(RST_SALES_QTY) AS SALES
    #              FROM (
    #                    SELECT DIVISION_CD
    #                         , SOLD_CUST_GRP_CD
    #                         , SKU_CD
    #                         , YY
    #                         , WEEK
    #                         , CASE WHEN UNIT_CD = 'BOX' THEN RST_SALES_QTY
    #                                WHEN UNIT_CD = 'EA' THEN ROUND(RST_SALES_QTY / BOX_EA, 2)
    #                                WHEN UNIT_CD = 'BOL' THEN ROUND(RST_SALES_QTY / BOX_BOL, 2)
    #                                ELSE 0
    #                                 END AS RST_SALES_QTY
    #                      FROM (
    #                            SELECT DIVISION_CD
    #                                 , SOLD_CUST_GRP_CD
    #                                 , SALES.SKU_CD
    #                                 , YY
    #                                 , WEEK
    #                                 , UNIT_CD
    #                                 , BOX_BOL
    #                                 , BOX_EA
    #                                 , RST_SALES_QTY
    #                              FROM (
    #                                    SELECT DIVISION_CD
    #                                         , SOLD_CUST_GRP_CD
    #                                         , ITEM_CD as SKU_CD
    #                                         , YY
    #                                         , WEEK
    #                                         , UNIT_CD
    #                                         , RST_SALES_QTY
    #                                     FROM  M4S_I002170_TEST
    #                                    WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
    #                                      AND RST_SALES_QTY > 0    -- Remove minus quantity
    #                                      --AND SOLD_CUST_GRP_CD = '1033' -- exception
    #                                  ) SALES
    #                      LEFT OUTER JOIN (
    #                                       SELECT BOX.ITEM_CD                                 AS SKU_CD
    #                                            , CONVERT(INT, BOX.FAC_PRICE / BOL.FAC_PRICE) AS BOX_BOL
    #                                            , CONVERT(INT, BOX.FAC_PRICE / EA.FAC_PRICE)  AS BOX_EA
    #                                         FROM (
    #                                               SELECT PROJECT_CD
    #                                                    , ITEM_CD
    #                                                    , PRICE_START_YYMMDD
    #                                                    , FAC_PRICE
    #                                                 FROM M4S_I002041
    #                                                WHERE PRICE_QTY_UNIT_CD = 'BOX'
    #                                                  AND FAC_PRICE <> 0
    #                                              ) BOX
    #                                         LEFT OUTER JOIN (
    #                                                          SELECT PROJECT_CD
    #                                                               , ITEM_CD
    #                                                               , PRICE_START_YYMMDD
    #                                                               , FAC_PRICE
    #                                                            FROM M4S_I002041
    #                                                           WHERE PRICE_QTY_UNIT_CD = 'BOL'
    #                                                             AND FAC_PRICE <> 0
    #                                                         ) BOL
    #                                           ON BOX.PROJECT_CD = BOL.PROJECT_CD
    #                                          AND BOX.ITEM_CD = BOL.ITEM_CD
    #                                          AND BOX.PRICE_START_YYMMDD = BOL.PRICE_START_YYMMDD
    #                                         LEFT OUTER JOIN (
    #                                                          SELECT PROJECT_CD
    #                                                               , ITEM_CD
    #                                                               , PRICE_START_YYMMDD
    #                                                               , FAC_PRICE
    #                                                            FROM M4S_I002041
    #                                                           WHERE PRICE_QTY_UNIT_CD = 'EA'
    #                                                             AND FAC_PRICE <> 0
    #                                                         ) EA
    #                                           ON BOX.PROJECT_CD = EA.PROJECT_CD
    #                                          AND BOX.ITEM_CD = EA.ITEM_CD
    #                                          AND BOX.PRICE_START_YYMMDD = EA.PRICE_START_YYMMDD
    #                                      ) UNIT
    #                        ON SALES.SKU_CD = UNIT.SKU_CD
    #                        ) SALES
    #                   ) SALES
    #              GROUP BY DIVISION_CD
    #                     , SOLD_CUST_GRP_CD
    #                     , SKU_CD
    #                     , YY
    #                     , WEEK
    #             """
    #     return sql

    # @staticmethod
    # def sql_sell_out_week_test(**kwargs):
    #     sql = f"""
    #         SELECT DIVISION_CD
    #              , CUST_GRP_CD
    #              , ITEM_ATTR01_CD AS BIZ_CD
    #              , ITEM_ATTR02_CD AS LINE_CD
    #              , ITEM_ATTR03_CD AS BRAND_CD
    #              , ITEM_ATTR04_CD AS ITEM_CD
    #              , MAP.ITEM_CD AS SKU_CD
    #              , YYMMDD
    #              , SEQ
    #              , DISCOUNT
    #              , WEEK
    #              , QTY
    #              , SELL.CREATE_DATE
    #           FROM (
    #                 SELECT DIVISION_CD
    #                      , SOLD_CUST_GRP_CD AS CUST_GRP_CD
    #                      , ITEM_CD AS SKU_CD
    #                      , YYMMDD
    #                      , SEQ
    #                      , DISCOUNT
    #                      , WEEK
    #                      , RST_SALES_QTY AS QTY
    #                      , CREATE_DATE
    #                   FROM M4S_I002173_SELL_OUT
    #                  WHERE SOLD_CUST_GRP_CD <> '1173'
    #                    AND YYMMDD BETWEEN {kwargs['from']} AND {kwargs['to']}
    #               ) SELL
    #          INNER JOIN M4S_I002179 MAP
    #             ON SELL.SKU_CD = MAP.BAR_CD
    #     """
    #     return sql

    # @staticmethod
    # def sql_sell_out_week_grp_test(**kwargs):
    #     sql = f"""
    #         SELECT DIVISION_CD
    #              , CUST_GRP_CD
    #              , SKU_CD
    #              , YY
    #              , WEEK
    #              , SUM(RST_SALES_QTY) AS SALES
    #           FROM (
    #                 SELECT DIVISION_CD
    #                      , SOLD_CUST_GRP_CD AS CUST_GRP_CD
    #                      , MAP.ITEM_CD          as SKU_CD
    #                      , YY
    #                      , WEEK
    #                      , RST_SALES_QTY
    #                   FROM (
    #                         SELECT *
    #                          FROM M4S_I002173_SELL_OUT
    #                         WHERE 1 = 1
    #                           AND SOLD_CUST_GRP_CD <> '1173'
    #                           AND YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
    #                        ) SALES
    #                  INNER JOIN (
    #                              SELECT BAR_CD
    #                                   , ITEM_CD
    #                                FROM M4S_I002179
    #                             ) MAP
    #                     ON SALES.ITEM_CD = MAP.BAR_CD
    #                ) SALES
    #         GROUP BY DIVISION_CD
    #                , CUST_GRP_CD
    #                , SKU_CD
    #                , YY
    #                , WEEK
    #     """
    #     return sql

    ##################
    # Previous Query
    ##################
    # SELL-OUT
    # @staticmethod
    # def sql_sell_out_week(**kwargs):
    #     sql = f"""
    #        SELECT DIVISION_CD
    #             , CUST_CD
    #             , BIZ_CD
    #             , LINE_CD
    #             , BRAND_CD
    #             , ITEM_CD
    #             , SALES.SKU_CD AS SKU_CD
    #             , YYMMDD
    #             , SEQ
    #             , DISCOUNT
    #             , WEEK
    #             , QTY
    #             , CREATE_DATE
    #          FROM (
    #                SELECT PROJECT_CD
    #                     , DIVISION_CD
    #                     , SOLD_CUST_GRP_CD AS CUST_CD
    #                     , ITEM_CD AS SKU_CD
    #                     , YYMMDD
    #                     , SEQ
    #                     , DISCOUNT
    #                     , WEEK
    #                     , RST_SALES_QTY AS QTY
    #                     , CREATE_DATE
    #                  FROM M4S_I002173
    #                 WHERE 1=1
    #                   AND RST_SALES_QTY <> 0
    #                -- WHERE YYMMDD BETWEEN {kwargs['from']} AND {kwargs['to']} # Todo: Exception
    #               ) SALES
    #         INNER JOIN (
    #                     SELECT ITEM_CD AS SKU_CD
    #                          , ITEM_ATTR01_CD AS BIZ_CD
    #                          , ITEM_ATTR02_CD AS LINE_CD
    #                          , ITEM_ATTR03_CD AS BRAND_CD
    #                          , ITEM_ATTR04_CD AS ITEM_CD
    #                       FROM VIEW_I002040
    #                      WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
    #                    ) ITEM
    #            ON SALES.SKU_CD = ITEM.SKU_CD
    #            """
    #     return sql

   # @staticmethod
    # def sql_sell_in(**kwargs):
    #     sql = f"""
    #         SELECT 'SELL_IN' AS DIVISION_CD
    #              , CUST_GRP_CD
    #              , BIZ_CD
    #              , LINE_CD
    #              , BRAND_CD
    #              , ITEM.ITEM_CD
    #              , SALES.SKU_CD
    #              , YYMMDD
    #              , SEQ
    #              , FROM_DC_CD
    #              , UNIT_PRICE
    #              , UNIT_CD
    #              , CASE WHEN FAC_PRICE IS NULL THEN 0
    #                     WHEN 1- (RST_SALES_PRICE * 100 / (QTY * FAC_PRICE)) < 0 THEN 0
    #                     ELSE 1- (RST_SALES_PRICE * 100 / (QTY * FAC_PRICE))
    #                      END AS DISCOUNT
    #              , WEEK
    #              , QTY
    #              , CREATE_DATE
    #           FROM (
    #                 SELECT PROJECT_CD
    #                      , DIVISION_CD
    #                      , SOLD_CUST_GRP_CD AS CUST_CD
    #                      , ITEM_CD AS SKU_CD
    #                      , YYMMDD
    #                      , SEQ
    #                      , FROM_DC_CD
    #                      , UNIT_PRICE
    #                      , UNIT_CD
    #                      , DISCOUNT
    #                      , WEEK
    #                      , RST_SALES_QTY AS QTY
    #                      , RST_SALES_PRICE
    #                      , CREATE_DATE
    #                   FROM M4S_I002170
    #                  WHERE YYMMDD BETWEEN {kwargs['from']} AND {kwargs['to']}
    #                    AND RST_SALES_QTY <> 0
    #                ) SALES
    #          INNER JOIN (
    #                      SELECT ITEM_CD AS SKU_CD
    #                           , ITEM_ATTR01_CD AS BIZ_CD
    #                           , ITEM_ATTR02_CD AS LINE_CD
    #                           , ITEM_ATTR03_CD AS BRAND_CD
    #                           , ITEM_ATTR04_CD AS ITEM_CD
    #                        FROM VIEW_I002040
    #                       WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
    #                         AND USE_YN = 'Y'
    #                         AND NEW_ITEM_YN = 'N'
    #                     ) ITEM
    #             ON SALES.SKU_CD = ITEM.SKU_CD
    #           LEFT OUTER JOIN (
    #                            SELECT CUST_CD
    #                                 , CUST_GRP_CD
    #                              FROM M4S_I002060
    #                           ) CUST
    #             ON SALES.CUST_CD = CUST.CUST_CD
    #           LEFT OUTER JOIN (
    #                            SELECT SKU_CD
    #                                 , FAC_PRICE
    #                              FROM (
    #                                    SELECT SKU_CD
    #                                         , PRICE_START_YYMMDD
    #                                         , FAC_PRICE
    #                                         , ROW_NUMBER() over (PARTITION BY SKU_CD
#   #                                           ORDER BY PRICE_START_YYMMDD DESC) AS SEQ
    #                                      FROM (
    #                                            SELECT ITEM_CD AS SKU_CD
    #                                                 , PRICE_START_YYMMDD
    #                                                 , FAC_PRICE
    #                                              FROM (
    #                                                    SELECT *
    #                                                      FROM M4S_I002041
    #                                                     WHERE PRICE_QTY_UNIT_CD = 'BOX'
    #                                                   ) PRICE
    #                                             GROUP BY ITEM_CD
    #                                                    , PRICE_START_YYMMDD
    #                                                    , FAC_PRICE
    #                                            ) RLST
    #                                   ) RLST
    #                             WHERE SEQ = 1
    #                           ) PRICE
    #             ON SALES.SKU_CD = PRICE.SKU_CD
    #            """
    #     return sql