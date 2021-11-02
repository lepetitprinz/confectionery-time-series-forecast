/* 실적데이터 */
SELECT PROJECT_CD
     , DIVISION_CD
     , CUST_GRP_CD
     , CUST_GRP_NM
     , CUST_CD
     , CUST_NM
     , ITEM_ATTR01_CD
     , ITEM_ATTR02_CD
     , ITEM_ATTR03_CD
     , ITEM_ATTR01_NM
     , ITEM_ATTR02_NM
     , ITEM_ATTR03_NM
     , ITEM_ATTR04_CD
     , ITEM_ATTR04_NM
     , ITEM_CD
     , ITEM_NM
     , WEEK
     , SEQ
     , YYMMDD
     , RST_SALES_QTY
  FROM (
        SELECT RESULT.PROJECT_CD
             , DIVISION_CD
             , CUST_GRP_CD
             , CUST_GRP_NM
             , RESULT.CUST_CD
             , CUST_NM
             , RESULT.ITEM_CD
             , ITEM_NM
             , ITEM_ATTR01_CD
             , ITEM_ATTR02_CD
             , ITEM_ATTR03_CD
             , ITEM_ATTR04_CD
             , ITEM_ATTR01_NM
             , ITEM_ATTR02_NM
             , ITEM_ATTR03_NM
             , ITEM_ATTR04_NM
             , WEEK
             , YYMMDD
             , SEQ
             , RST_SALES_QTY
          FROM (
               SELECT PROJECT_CD
                    , DIVISION_CD
                    , SOLD_CUST_GRP_CD AS CUST_CD
                    , ITEM_CD
                    , YYMMDD
                    , WEEK
                    , SEQ
                    , RST_SALES_QTY
                 FROM M4S_I002170
                WHERE 1=1
--                   AND YYMMDD BETWEEN (SELECT OPTION_VAL FROM M4S_I001020 WHERE OPTION_CD = 'RST_START_DAY')
--                                  AND (SELECT OPTION_VAL FROM M4S_I001020 WHERE OPTION_CD = 'RST_END_DAY')
               ) RESULT
          LEFT OUTER JOIN (
                           SELECT ITEM_CD
                                , ITEM_NM
                                , ITEM_ATTR01_CD
                                , ITEM_ATTR01_NM
                                , ITEM_ATTR02_CD
                                , ITEM_ATTR02_NM
                                , ITEM_ATTR03_CD
                                , ITEM_ATTR03_NM
                                , ITEM_ATTR04_CD
                                , ITEM_ATTR04_NM
                             FROM VIEW_I002040
                            WHERE ITEM_TYPE_CD IN ('FERT', 'HAWA')
                            GROUP BY ITEM_CD
                                   , ITEM_NM
                                   , ITEM_ATTR01_CD
                                   , ITEM_ATTR01_NM
                                   , ITEM_ATTR02_CD
                                   , ITEM_ATTR02_NM
                                   , ITEM_ATTR03_CD
                                   , ITEM_ATTR03_NM
                                   , ITEM_ATTR04_CD
                                   , ITEM_ATTR04_NM
                          ) COMM
            ON RESULT.ITEM_CD = COMM.ITEM_CD
          LEFT OUTER JOIN (
                           SELECT CUST.CUST_GRP_CD
                                , GRP.CUST_GRP_NM
                                , CUST.CUST_CD
                                , CUST_NM
                            FROM (
                                  SELECT CUST_GRP_CD
                                       , CUST_CD
                                       , CUST_NM
                                    FROM M4S_I002060
                                   WHERE 1 = 1
                                     AND USE_YN = 'Y'
                                 ) CUST
                            LEFT OUTER JOIN (
                                             SELECT CUST_GRP_CD
                                                  , CUST_GRP_NM
                                             FROM M4S_I002050
                                             WHERE 1 = 1
                                               AND USE_YN = 'Y'
                                            ) GRP
                              ON CUST.CUST_GRP_CD = GRP.CUST_GRP_CD
              ) CUST
         ON RESULT.CUST_CD = CUST.CUST_CD
        WHERE 1 = 1
--           AND (ITEM_ATTR02_CD = 'P111' OR ITEM_ATTR03_CD = 'P304020') -- PROTOTYPE 임시 처리
        ) RSLT