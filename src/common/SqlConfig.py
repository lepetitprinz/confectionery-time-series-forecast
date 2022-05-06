
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
                 , ITEM1.NAME AS BIZ_NM
                 , ITEM_ATTR02_CD AS LINE_CD
                 , ITEM2.NAME AS LINE_NM
                 , ITEM_ATTR03_CD AS BRAND_CD
                 , ITEM3.NAME AS BRAND_NM
                 , ITEM_ATTR04_CD AS ITEM_CD
                 , ITEM4.NAME AS ITEM_NM
                 , ITEM_CD AS SKU_CD
                 , ITEM_NM AS SKU_NM
              FROM (
                    SELECT ITEM_ATTR01_CD
                         , ITEM_ATTR02_CD
                         , ITEM_ATTR03_CD
                         , ITEM_ATTR04_CD
                         , ITEM_CD
                         , ITEM_NM
                      FROM M4S_I002040
                     WHERE ITEM_TYPE_CD IN ('FERT', 'HAWA')
                       AND USE_YN = 'Y'
                       AND DEL_YN = 'N'
                   ) ITEM
              LEFT OUTER JOIN (
                               SELECT COMM_DTL_CD AS CODE
                                    , COMM_DTL_CD_NM AS NAME
                                 FROM M4S_I002011
                                WHERE COMM_CD = 'ITEM_GUBUN01'
                              ) ITEM1
                ON ITEM.ITEM_ATTR01_CD = ITEM1.CODE
              LEFT OUTER JOIN (
                               SELECT COMM_DTL_CD AS CODE
                                    , COMM_DTL_CD_NM AS NAME
                                 FROM M4S_I002011
                                WHERE COMM_CD = 'ITEM_GUBUN02'
                              ) ITEM2
                ON ITEM.ITEM_ATTR02_CD = ITEM2.CODE
              LEFT OUTER JOIN (
                               SELECT COMM_DTL_CD AS CODE
                                    , COMM_DTL_CD_NM AS NAME
                                 FROM M4S_I002011
                                WHERE COMM_CD = 'ITEM_GUBUN03'
                              ) ITEM3
                ON ITEM.ITEM_ATTR03_CD = ITEM3.CODE
              LEFT OUTER JOIN (
                               SELECT COMM_DTL_CD AS CODE
                                    , COMM_DTL_CD_NM AS NAME
                                 FROM M4S_I002011
                                WHERE COMM_CD = 'ITEM_GUBUN04'
                              ) ITEM4
                ON ITEM.ITEM_ATTR04_CD = ITEM4.CODE
        """
        return sql

    @staticmethod
    def sql_item_mega_yn():
        sql = """
            SELECT ITEM_CD
                 , MEGA_YN
              FROM M4S_I002040
             WHERE ITEM_TYPE_CD IN ('FERT', 'HAWA')
               AND USE_YN = 'Y'
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
            SELECT BOM.SKU_CD AS SKU_CD
                 , ITEM.ITEM_NM AS SKU_NM
                 , BOM_CD
                 , ING.ITEM_NM AS BOM_NM
              FROM (
                    SELECT PROD_ITEM_CD AS SKU_CD
                         , CNSM_ITEM_CD AS BOM_CD
                      FROM M4S_I305200
                     GROUP BY PROD_ITEM_CD
                            , CNSM_ITEM_CD
                   ) BOM
             INNER JOIN (
                         SELECT ITEM_CD
                              , ITEM_NM
                           FROM M4S_I002040
                          WHERE ITEM_TYPE_CD IN ('HAWA', 'FERT')
                        ) ITEM
                ON BOM.SKU_CD = ITEM.ITEM_CD
             INNER JOIN (
                         SELECT ITEM_CD
                              , ITEM_NM
                           FROM M4S_I002040
                        ) ING
                ON BOM.BOM_CD = ING.ITEM_CD
        """
        return sql

    @staticmethod
    def sql_bom_mst_bak():
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
               AND STAT_CD <> 'VOTING'  -- Exception           
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
    def sql_sales_matrix():
        sql = """
             SELECT RIGHT(SALES_MGMT_CD, 4) AS CUST_GRP_CD
                  , ITEM_CD AS SKU_CD
               FROM M4S_I204050
              WHERE USE_YN = 'Y'
                AND SALES_MGMT_VRSN_ID = (
                                          SELECT SALES_MGMT_VRSN_ID 
                                            FROM M4S_I204010
                                           WHERE USE_YN = 'Y'
                                         )
        """
        return sql

    ######################################
    # Sales dataset
    ######################################
    # SELL-IN Table
    # @staticmethod
    # def sql_sell_in(**kwargs):
    #     sql = f"""
    #          SELECT DIVISION_CD
    #              , CUST_GRP_CD
    #              , ITEM_ATTR01_CD AS BIZ_CD
    #              , ITEM_ATTR02_CD AS LINE_CD
    #              , ITEM_ATTR03_CD AS BRAND_CD
    #              , ITEM_ATTR04_CD AS ITEM_CD
    #              , ITEM_CD AS SKU_CD
    #              , YYMMDD
    #              , SEQ
    #              , FROM_DC_CD
    #              , UNIT_PRICE
    #              , UNIT_CD
    #              , DISCOUNT
    #              , WEEK
    #              , QTY
    #              , CREATE_DATE
    #           FROM M4S_I002176
    #          WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
    #     """
    #     return sql

    @staticmethod
    def sql_sell_in(**kwargs):
        sql = f"""
             SELECT DIVISION_CD
                  , SALES.CUST_GRP_CD
                  , ITEM_ATTR01_CD AS BIZ_CD
                  , ITEM_ATTR02_CD AS LINE_CD
                  , ITEM_ATTR03_CD AS BRAND_CD
                  , ITEM_ATTR04_CD AS ITEM_CD
                  , SALES.ITEM_CD AS SKU_CD
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
                     SELECT *
                       FROM M4S_I002176
                      WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                        AND QTY <> 0
                    ) SALES
              INNER JOIN (
                          SELECT RIGHT(SALES_MGMT_CD, 4) AS CUST_GRP_CD
                               , ITEM_CD
                            FROM M4S_I204050
                           WHERE USE_YN = 'Y'
                             AND SALES_MGMT_VRSN_ID = (
                                                       SELECT SALES_MGMT_VRSN_ID 
                                                         FROM M4S_I204010 
                                                        WHERE USE_YN = 'Y'
                                                      )
                         ) MAP
                 ON SALES.CUST_GRP_CD = MAP.CUST_GRP_CD
                AND SALES.ITEM_CD = MAP.ITEM_CD
        """
        return sql

    @staticmethod
    def sql_sell_in_week_grp(**kwargs):
        sql = f""" 
            SELECT DIVISION_CD
                 , SALES.CUST_GRP_CD
                 , SALES.ITEM_CD AS SKU_CD
                 , YYMMDD AS YY
                 , WEEK
                 , RST_SALES_QTY AS SALES
            FROM (
                  SELECT *
                    FROM M4S_I002175
                   WHERE RST_SALES_QTY <> 0
                     AND DIVISION_CD = 'SELL_IN'
                     AND START_WEEK_DAY BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                 ) SALES
           INNER JOIN (
                       SELECT RIGHT(SALES_MGMT_CD, 4) AS CUST_GRP_CD
                            , ITEM_CD
                         FROM M4S_I204050
                        WHERE USE_YN = 'Y'
                          AND SALES_MGMT_VRSN_ID = (
                                                    SELECT SALES_MGMT_VRSN_ID 
                                                      FROM M4S_I204010 
                                                     WHERE USE_YN = 'Y'
                                                   )
                      ) MAP
              ON SALES.CUST_GRP_CD = MAP.CUST_GRP_CD
             AND SALES.ITEM_CD = MAP.ITEM_CD
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
    def sql_err_grp_map():
        sql = """
            SELECT COMM_DTL_CD
                 , ATTR01_VAL
              FROM M4S_I002011
             WHERE COMM_CD = 'ERR_CD'
            """
        return sql

    @staticmethod
    def sql_exg_data(**kwargs):
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
                                                 AND PARTIAL_YN = '{kwargs['partial_yn']}'
                                               )
                             )
              AND YYMM BETWEEN {kwargs['from']} AND {kwargs['to']}
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
                               AND RST_SALES_QTY <> 0
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
            -- AND ITEM_NM NOT LIKE N'%삭제%'
        """
        return sql

    @staticmethod
    def sql_pred_best(**kwargs):
        sql = f"""
            SELECT DIVISION_CD
                 , CUST_GRP_CD
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , ITEM_CD
                 --, STAT_CD
                 , YYMMDD AS START_WEEK_DAY
                 , WEEK
                 , RESULT_SALES AS PRED
             FROM M4S_O110600
            WHERE DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
              AND DIVISION_CD = '{kwargs['division_cd']}'
              AND LEFT(FKEY, 5) = '{kwargs['fkey']}'
              AND YYMMDD = '{kwargs['yymmdd']}'
        """
        return sql

    @staticmethod
    def sql_sell_week_compare(**kwargs):
        sql = f"""
            SELECT CUST_GRP_CD
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , ITEM_CD
                 , START_WEEK_DAY
                 , WEEK
                 , IIF(RST_SALES_QTY < 0, 0, RST_SALES_QTY) AS SALES
              FROM M4S_I002175
             WHERE DIVISION_CD = '{kwargs['division_cd']}'
               AND START_WEEK_DAY = '{kwargs['start_week_day']}'
               AND RST_SALES_QTY <> 0
        """
        return sql

    @staticmethod
    def sql_sell_week_hist(**kwargs):
        sql = f"""
            SELECT DIVISION_CD
                 , SALES.CUST_GRP_CD
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , SALES.ITEM_CD
                 , START_WEEK_DAY
                 , WEEK
                 , SALES
              FROM (
                    SELECT DIVISION_CD
                         , CUST_GRP_CD
                         , ITEM_ATTR01_CD
                         , ITEM_ATTR02_CD
                         , ITEM_ATTR03_CD
                         , ITEM_ATTR04_CD
                         , ITEM_CD
                         , START_WEEK_DAY
                         , WEEK
                         , RST_SALES_QTY AS SALES
                      FROM M4S_I002175 SALES
                     WHERE DIVISION_CD = '{kwargs['division_cd']}'
                       AND START_WEEK_DAY BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                       AND RST_SALES_QTY <> 0 
                    ) SALES
             INNER JOIN (
                         SELECT RIGHT(SALES_MGMT_CD, 4) AS CUST_GRP_CD
                              , ITEM_CD
                           FROM M4S_I204050
                          WHERE USE_YN = 'Y'
                            AND SALES_MGMT_VRSN_ID = (
                                                      SELECT SALES_MGMT_VRSN_ID 
                                                        FROM M4S_I204010 
                                                       WHERE USE_YN = 'Y'
                                                     )
                        ) MAP
                ON SALES.CUST_GRP_CD = MAP.CUST_GRP_CD
               AND SALES.ITEM_CD = MAP.ITEM_CD
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
               AND CUST_GRP_CD IN {kwargs['cust_grp_cd']}
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
               AND CUST_GRP_CD IN {kwargs['cust_grp_cd']}
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
               AND CUST_GRP_CD IN {kwargs['cust_grp_cd']}
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
    def del_pred_recent(**kwargs):
        sql = f"""
            DELETE
             FROM M4S_O111600
            WHERE DIVISION_CD = '{kwargs['division_cd']}'
              AND CUST_GRP_CD IN {kwargs['cust_grp_cd']}
        """
        return sql

    @staticmethod
    def del_pred_plan_acc(**kwargs):
        sql = f"""
            DELETE
              FROM M4S_O110630
             WHERE DATA_VRSN_CD = '{kwargs['data_vrsn_cd']}'
               AND DIVISION_CD = '{kwargs['division_cd']}'
               AND YYMMDD = '{kwargs['yymmdd']}'

        """
        return sql

    # --------------------------------------------------

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
                 , N'전국' AS IDX_DTL_NM
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

    @staticmethod
    def sql_cust_sp1_map_error(**kwargs):
        sql = f"""
            SELECT RSLT.CUST_CD
                 , CUST_NM
                 , SUBSTRING(CUST_MAP, 1, 3) AS CUST_ORG_CD
                 , SUBSTRING(CUST_MAP, 5, 2) AS ITEM_CLASS_CD
                 , SUBSTRING(CUST_MAP, 8, 2) AS CUST_DIST_CD
              FROM (
                    SELECT CUST_CD
                         , CUST_GRP_CD
                         , CUST_MAP
                      FROM (
                            SELECT SALES.CUST_CD
                                 , SP1_CD AS CUST_GRP_CD
                                 , SALES.CUST_MAP
                              FROM (
                                    SELECT SOLD_CUST_GRP_CD AS CUST_CD
                                         , ITEM_CD          AS SKU_CD
                                         , SEQ
                                         , LEFT(SEQ, 9)     AS CUST_MAP
                                         , RST_SALES_QTY    AS QTY
                                      FROM M4S_I002170
                                     WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                                   ) SALES
                              INNER JOIN (
                                               SELECT CUST_CD
                                                    , CONCAT(CUST_ORG_CD, '_', ITEM_CLASS_CD, '_', CUST_DIST_CD) AS CUST_MAP
                                                    , SP1_CD
                                                 FROM M4S_I002061
                                                WHERE ITEM_CLASS_CD IN ('10', '20', '60')
                                              ) CUST
                                ON SALES.CUST_CD = CUST.CUST_CD
                               AND SALES.CUST_MAP = CUST.CUST_MAP
                           ) RSLT
                     WHERE CUST_GRP_CD IS NULL
                     GROUP BY CUST_CD
                            , CUST_GRP_CD
                            , CUST_MAP
                   ) RSLT
              LEFT OUTER JOIN (
                               SELECT CUST_CD
                                    , CUST_NM
                                 FROM M4S_I002060
                              ) CUST_MST
                ON RSLT.CUST_CD = CUST_MST.CUST_CD
             ORDER BY CUST_CD
        """

        return sql

    @staticmethod
    def sql_acc_by_sp1(**kwargs):
        sql = f"""
            SELECT ACC_GRP
                 , COUNT
                 , SUM(COUNT) OVER(ORDER BY ACC_GRP) AS CUM_COUNT
              FROM (
                    SELECT ACC_GRP
                         , COUNT(ACC_GRP) AS COUNT
                      FROM (
                            SELECT CASE WHEN ACCURACY BETWEEN 0.75 AND 1.25 THEN 'A'
                                        WHEN ACCURACY BETWEEN 0.6 AND 1.4 THEN 'B'
                                        WHEN ACCURACY BETWEEN 0.5 AND 1.5 THEN 'C'
                                        ELSE 'F'
                                         END AS ACC_GRP
                              FROM (
                                    SELECT CASE WHEN SALES = PRED THEN 1
                                                -- WHEN SALES = 0 THEN 0
                                                WHEN PRED = 0 THEN 0
                                                ELSE SALES / PRED
                                                 END AS ACCURACY
                                      FROM (
                                            SELECT ISNULL(SALES.SALES, 0) AS SALES
                                                 , PRED.PRED
                                              FROM (
                                                    SELECT DIVISION_CD
                                                         , CUST_GRP_CD
                                                         , ITEM_CD
                                                         , RESULT_SALES AS PRED
                                                         , YYMMDD
                                                      FROM M4S_O110600
                                                     WHERE DIVISION_CD = '{kwargs['division']}'
                                                       AND DATA_VRSN_CD = '{kwargs['data_vrsn']}'
                                                       AND CUST_GRP_CD = '{kwargs['sp1']}'
                                                       AND LEFT(FKEY, 5) = 'C1-P5'
                                                       AND YYMMDD = '{kwargs['yymmdd']}'
                                                       AND ITEM_CD IN (
                                                                       SELECT ITEM_CD
                                                                         FROM (
                                                                               SELECT ITEM_ATTR01_CD
                                                                                    , ITEM_CD
                                                                                    , AVG(RST_SALES_QTY) AS AVG_QTY
                                                                                 FROM (
                                                                                       SELECT CUST_GRP_CD
                                                                                            , ITEM_ATTR01_CD
                                                                                            , ITEM_CD
                                                                                            , RST_SALES_QTY
                                                                                         FROM M4S_I002175
                                                                                        WHERE DIVISION_CD = '{kwargs['division']}'
                                                                                          AND CUST_GRP_CD = '{kwargs['sp1']}'
                                                                                          AND RST_SALES_QTY > 0
                                                                                          AND START_WEEK_DAY = '{kwargs['yymmdd_comp']}'
                                                                                      ) SALES
                                                                                GROUP BY CUST_GRP_CD
                                                                                       , ITEM_ATTR01_CD
                                                                                       , ITEM_CD
                                                                              ) RSLT
                                                                        WHERE AVG_QTY >= {kwargs['threshold']}
                                                                          AND RSLT.ITEM_ATTR01_CD = '{kwargs['biz']}'
                                                                      )
                                                   ) PRED
                                              LEFT OUTER JOIN (
                                                               SELECT DIVISION_CD
                                                                    , CUST_GRP_CD
                                                                    , ITEM_CD
                                                                    , RST_SALES_QTY  AS SALES
                                                                    , START_WEEK_DAY AS YYMMDD
                                                                 FROM M4S_I002175
                                                                WHERE DIVISION_CD = '{kwargs['division']}'
                                                                  AND CUST_GRP_CD = '{kwargs['sp1']}'
                                                                  AND START_WEEK_DAY = '{kwargs['yymmdd_comp']}'
                                                              ) SALES
                                                ON PRED.DIVISION_CD = SALES.DIVISION_CD
                                               AND PRED.CUST_GRP_CD = SALES.CUST_GRP_CD
                                               AND PRED.ITEM_CD = SALES.ITEM_CD
                                           ) RSLT
                                     WHERE SALES <> 0
                                   ) RSLT
                           ) RSLT
                     GROUP BY ACC_GRP
                 ) CNT
        
        """

        return sql

    @staticmethod
    def sql_acc_by_line(**kwargs):
        sql = f"""
            SELECT ITEM_ATTR02_CD
                 , ROUND(ACC_S * 1.0 / (ACC_S + ACC_F), 2) AS RESULT
              FROM (
                    SELECT ITEM_ATTR02_CD
                         , ISNULL(SUM(CASE WHEN ACC_GRP = 'S' THEN 1 END), 0) AS ACC_S
                         , ISNULL(SUM(CASE WHEN ACC_GRP = 'F' THEN 1 END), 0) AS ACC_F
                      FROM (
                            SELECT ITEM_ATTR02_CD
                                 , IIF(ACCURACY BETWEEN 0.5 AND 1.5, 'S', 'F') AS ACC_GRP
                              FROM (
                                    SELECT ITEM_ATTR02_CD
                                         , CASE WHEN SALES = PRED THEN 1
                                                WHEN PRED = 0 THEN 0
    --                                          ELSE PRED / SALES
                                           ELSE SALES / PRED
                                            END AS ACCURACY
                                      FROM (
                                            SELECT PRED.ITEM_ATTR02_CD
                                                 , ISNULL(SALES.SALES, 0) AS SALES
                                                 , PRED.PRED
                                              FROM (
                                                    SELECT DIVISION_CD
                                                         , CUST_GRP_CD
                                                         , ITEM_ATTR02_CD
                                                         , ITEM_CD
                                                         , RESULT_SALES AS PRED
                                                         , YYMMDD
                                                      FROM M4S_O110600
                                                     WHERE DIVISION_CD = '{kwargs['division']}'
                                                       AND DATA_VRSN_CD = '{kwargs['data_vrsn']}'
                                                       AND CUST_GRP_CD = '{kwargs['sp1']}'
                                                       AND LEFT(FKEY, 5) = 'C1-P5'
                                                       AND YYMMDD = '{kwargs['yymmdd']}'
                                                       AND ITEM_CD IN (
                                                                        SELECT ITEM_CD
                                                                         FROM (
                                                                               SELECT ITEM_ATTR01_CD
                                                                                    , ITEM_CD
                                                                                    , AVG(RST_SALES_QTY) AS AVG_QTY
                                                                                 FROM (
                                                                                       SELECT CUST_GRP_CD
                                                                                            , ITEM_ATTR01_CD
                                                                                            , ITEM_CD
                                                                                            , RST_SALES_QTY
                                                                                         FROM M4S_I002175
                                                                                        WHERE DIVISION_CD = '{kwargs['division']}'
                                                                                          AND CUST_GRP_CD = '{kwargs['sp1']}'
                                                                                          AND RST_SALES_QTY > 0
                                                                                          AND START_WEEK_DAY = '{kwargs['yymmdd_comp']}'
                                                                                      ) SALES
                                                                                GROUP BY CUST_GRP_CD
                                                                                       , ITEM_ATTR01_CD
                                                                                       , ITEM_CD
                                                                              ) RSLT
                                                                        WHERE AVG_QTY >= {kwargs['threshold']}
                                                                          AND RSLT.ITEM_ATTR01_CD = '{kwargs['biz']}'
                                                                      )
                                                   ) PRED
                                              LEFT OUTER JOIN (
                                                               SELECT DIVISION_CD
                                                                    , CUST_GRP_CD
                                                                    , ITEM_ATTR02_CD
                                                                    , ITEM_CD
                                                                    , RST_SALES_QTY  AS SALES
                                                                    , START_WEEK_DAY AS YYMMDD
                                                                 FROM M4S_I002175
                                                                WHERE DIVISION_CD = '{kwargs['division']}'
                                                                  AND CUST_GRP_CD = '{kwargs['sp1']}'
                                                                  AND START_WEEK_DAY = '{kwargs['yymmdd_comp']}'
                                                               ) SALES
                                                ON PRED.DIVISION_CD = SALES.DIVISION_CD
                                               AND PRED.CUST_GRP_CD = SALES.CUST_GRP_CD
                                               AND PRED.ITEM_CD = SALES.ITEM_CD
                                               AND PRED.ITEM_ATTR02_CD = SALES.ITEM_ATTR02_CD
                                           ) RSLT
                                     WHERE SALES <> 0
                                    ) RSLT
                             ) RSLT
                    GROUP BY ITEM_ATTR02_CD
                ) RSLT """

        return sql

    @staticmethod
    def sql_sales_plan_confirm(**kwargs):
        sql = f"""
            SELECT SP2_C_CD
                 , SP2_CD
                 , SP1_C_CD
                 , CUST_GRP_CD
                 , START_WEEK_DAY
                 , WEEK
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , PLANED.ITEM_CD
                 , PLANED
                 , MEGA_YN
              FROM (
                    SELECT SP2_C_CD
                         , SP2_CD
                         , SP1_C_CD
                         , SP1_CD AS CUST_GRP_CD
                         , YYMMDD AS START_WEEK_DAY
                         , WEEK
                         , ITEM_ATTR01_CD
                         , ITEM_ATTR02_CD
                         , ITEM_ATTR03_CD
                         , ITEM_ATTR04_CD
                         , ITEM_CD
                         , SP1_QTY AS PLANED
                      FROM M4S_O202020
                     WHERE YYMMDD = '{kwargs['yymmdd']}'
                    ) PLANED
             INNER JOIN (
                         SELECT ITEM_CD
                              , MEGA_YN
                           FROM M4S_I002040
                        ) ITEM
                ON PLANED.ITEM_CD = ITEM.ITEM_CD
            """
        return sql

    @staticmethod
    def sql_cust_nm_master():
        sql = """
            SELECT SALES_MGMT_CD AS CODE
                 , SALES_MGMT_NM AS NAME
                 , SALES_MGMT_TYPE_CD AS TYPE
              FROM M4S_I204030
             WHERE SALES_MGMT_TYPE_CD = 'SP2_C'
               AND SALES_MGMT_VRSN_ID = (
                                         SELECT SALES_MGMT_VRSN_ID
                                           FROM M4S_I204010
                                          WHERE USE_YN = 'Y'
                                        )
             UNION
            SELECT RIGHT(SALES_MGMT_CD, 2) AS SALES_MGMT_CD
                 , SALES_MGMT_NM
                 , SALES_MGMT_TYPE_CD
              FROM M4S_I204030
            WHERE SALES_MGMT_TYPE_CD = 'SP2'
              AND SALES_MGMT_VRSN_ID = (
                                        SELECT SALES_MGMT_VRSN_ID
                                          FROM M4S_I204010
                                         WHERE USE_YN = 'Y'
                                        )
             UNION
            SELECT RIGHT(SALES_MGMT_CD, 3) AS SALES_MGMT_CD
                 , SALES_MGMT_NM
                 , SALES_MGMT_TYPE_CD
              FROM M4S_I204030
             WHERE SALES_MGMT_TYPE_CD = 'SP1_C'
               AND SALES_MGMT_VRSN_ID = (
                                         SELECT SALES_MGMT_VRSN_ID
                                           FROM M4S_I204010
                                          WHERE USE_YN = 'Y'
                                       )
             GROUP BY RIGHT(SALES_MGMT_CD, 3)
                    , SALES_MGMT_NM
                    , SALES_MGMT_TYPE_CD
            UNION
           SELECT RIGHT(SALES_MGMT_CD, 4) AS SALES_MGMT_CD
                , SALES_MGMT_NM
                , SALES_MGMT_TYPE_CD
             FROM M4S_I204030
            WHERE SALES_MGMT_TYPE_CD = 'SP1'
              AND SALES_MGMT_VRSN_ID = (
                                        SELECT SALES_MGMT_VRSN_ID
                                          FROM M4S_I204010
                                         WHERE USE_YN = 'Y'
                                       )
            GROUP BY RIGHT(SALES_MGMT_CD, 4)
                   , SALES_MGMT_NM
                   , SALES_MGMT_TYPE_CD
        """
        return sql

    @staticmethod
    def sql_pred_plan_sell_in(**kwargs):
        sql = f"""
            SELECT SP2_C_CD
                 , SP2_CD
                 , SP1_C_CD
                 , CUST_GRP_CD
                 , START_WEEK_DAY
                 , WEEK
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , COMPARE.ITEM_CD
                 , ISNULL(SALES, 0) AS SALES
                 , ISNULL(PRED, 0) AS PRED
                 , ISNULL(PLANED, 0) AS PLANED
                 , MEGA_YN
              FROM (
                    SELECT SP2_C_CD
                         , SP2_CD
                         , SP1_C_CD
                         , SP1_CD AS CUST_GRP_CD
                         , YYMMDD AS START_WEEK_DAY
                         , WEEK
                         , ITEM_ATTR01_CD
                         , ITEM_ATTR02_CD
                         , ITEM_ATTR03_CD
                         , ITEM_ATTR04_CD
                         , ITEM_CD
                         , RST_IN_QTY AS SALES
                         , DF_IN_QTY AS PRED
                         , SP1_QTY AS PLANED
                      FROM M4S_O202020
                     WHERE YYMMDD = '{kwargs['yymmdd']}'
                    ) COMPARE
             INNER JOIN (
                         SELECT ITEM_CD
                              , MEGA_YN
                           FROM M4S_I002040
                        ) ITEM
                ON COMPARE.ITEM_CD = ITEM.ITEM_CD
        """
        return sql

    @staticmethod
    def sql_pred_plan_sell_out(**kwargs):
        sql = f"""
            SELECT SP2_C_CD
                 , SP2_CD
                 , SP1_C_CD
                 , CUST_GRP_CD
                 , START_WEEK_DAY
                 , WEEK
                 , ITEM_ATTR01_CD
                 , ITEM_ATTR02_CD
                 , ITEM_ATTR03_CD
                 , ITEM_ATTR04_CD
                 , COMPARE.ITEM_CD
                 , ISNULL(SALES, 0) AS SALES
                 , ISNULL(PRED, 0) AS PRED
                 , ISNULL(PLANED, 0) AS PLANED
                 , MEGA_YN
              FROM (
                    SELECT SP2_C_CD
                         , SP2_CD
                         , SP1_C_CD
                         , SP1_CD AS CUST_GRP_CD
                         , YYMMDD AS START_WEEK_DAY
                         , WEEK
                         , ITEM_ATTR01_CD
                         , ITEM_ATTR02_CD
                         , ITEM_ATTR03_CD
                         , ITEM_ATTR04_CD
                         , ITEM_CD
                         , RST_OUT_QTY AS SALES
                         , DF_OUT_QTY AS PRED
                         , SP1_QTY AS PLANED
                      FROM M4S_O202020
                     WHERE YYMMDD = '{kwargs['yymmdd']}'
                       AND SP1_CD IN ('1065', '1066', '1067', '1073', '1074', '1075', '1076')
                    ) COMPARE
             INNER JOIN (
                         SELECT ITEM_CD
                              , MEGA_YN
                           FROM M4S_I002040
                        ) ITEM
                ON COMPARE.ITEM_CD = ITEM.ITEM_CD
        """
        return sql

    @staticmethod
    def sql_sell_in_dist(**kwargs):
        sql = f"""
             SELECT SALES.CUST_GRP_CD
                  , ITEM_ATTR01_CD AS BIZ_CD
                  , ITEM_ATTR02_CD AS LINE_CD
                  , ITEM_ATTR03_CD AS BRAND_CD
                  , ITEM_ATTR04_CD AS ITEM_CD
                  , SALES.ITEM_CD AS SKU_CD
                  , YYMMDD
                  , QTY AS IN_QTY
               FROM (
                     SELECT CUST_GRP_CD
                          , ITEM_ATTR01_CD
                          , ITEM_ATTR02_CD 
                          , ITEM_ATTR03_CD
                          , ITEM_ATTR04_CD 
                          , ITEM_CD
                          , YYMMDD
                          , SUM(QTY) AS QTY
                       FROM M4S_I002176
                      WHERE YYMMDD BETWEEN '{kwargs['from']}' AND '{kwargs['to']}'
                        AND QTY <> 0
                        AND CUST_GRP_CD IN ('1065', '1066', '1067', '1073', '1074', '1075', '1076')
                      GROUP BY CUST_GRP_CD
                          , ITEM_ATTR01_CD
                          , ITEM_ATTR02_CD 
                          , ITEM_ATTR03_CD
                          , ITEM_ATTR04_CD 
                          , ITEM_CD
                          , YYMMDD
                    ) SALES
              INNER JOIN (
                          SELECT RIGHT(SALES_MGMT_CD, 4) AS CUST_GRP_CD
                               , ITEM_CD
                            FROM M4S_I204050
                           WHERE USE_YN = 'Y'
                             AND SALES_MGMT_VRSN_ID = (
                                                       SELECT SALES_MGMT_VRSN_ID 
                                                         FROM M4S_I204010 
                                                        WHERE USE_YN = 'Y'
                                                      )
                         ) MAP
                 ON SALES.CUST_GRP_CD = MAP.CUST_GRP_CD
                AND SALES.ITEM_CD = MAP.ITEM_CD
        """
        return sql

    @staticmethod
    def sql_sp1_sp1c(sp1_list: tuple):
        sql = f"""
          SELECT RIGHT(PARENT_SALES_MGMT_CD, 3) as SP1C_CD
               , LINK_SALES_MGMT_CD as CUST_GRP_CD
            FROM M4S_I204030
           WHERE USE_YN = 'Y'
             AND SALES_MGMT_TYPE_CD = 'SP1'
             AND SALES_MGMT_VRSN_ID = (SELECT SALES_MGMT_VRSN_ID FROM M4S_I204010 WHERE USE_YN = 'Y')
             AND RIGHT(PARENT_SALES_MGMT_CD, 3) IN {sp1_list}
           GROUP BY RIGHT(PARENT_SALES_MGMT_CD, 3)
               , LINK_SALES_MGMT_CD
            """
        return sql
