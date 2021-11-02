SELECT 'ENT001' AS PROJECT_CD
     , 'SELL_OUT' AS DIVISION_CD
     , '1067' AS SOLD_CUST_GRP_CD
     , '1067' AS SHIP_CUST_GRP_CD
     , ITEM_CD
     , SALES.YYMMDD
     , CONCAT(SALES.YYMMDD, '-', FORMAT(ROW_NUMBER() OVER (PARTITION BY SALES.YYMMDD ORDER BY SALES.YYMMDD), '000000')) AS SEQ
     , DISCOUNT
     , WEEK
     , PART_WEEK
     , YYMM
     , YY
     , NULL AS USER_CD
     , RST_SALES_QTY
     , RST_SALES_PRICE
     , 'SYSTEM' AS CREATE_USER_CD
     , GETDATE() AS CREATE_DATE
     , NULL AS MODIFY_USER_CD
     , NULL AS MODIFY_DATE
  FROM (
        SELECT MST.SELL_BARCD AS ITEM_CD
             , SELL_DATE AS YYMMDD
             , SELL_QTY   AS RST_SALES_QTY
             , SELL_AMT AS RST_SALES_PRICE
             , SELL_PRICE AS DISCOUNT
          FROM (
                SELECT *
                  FROM TEST.dbo.SELLOUT_PAST_TEMP SALES
                 WHERE 1=1
                   AND CHAIN_CODE = '60001'
                   AND SELL_QTY <> 0
               ) SELL
          LEFT OUTER JOIN (
                           SELECT SELL_BARCD
                                , PROD_NM
                             FROM TEST.dbo.SELLOUT_PAST_HOME_MASTER
                          ) MST
            ON SELL.PROD_NM = MST.PROD_NM
         WHERE MST.SELL_BARCD IS NOT NULL
       ) SALES
  LEFT OUTER JOIN (
                   SELECT YYMMDD
                        , WEEK
                        , PART_WEEK
                        , YYMM
                        , YY
                     FROM M4S_I002030
                  ) CAL
    ON SALES.YYMMDD = CAL.YYMMDD