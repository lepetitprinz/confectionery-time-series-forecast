SELECT 'ENT001' AS PROJECT_CD
     , 'SELL_OUT' AS DIVISION_CD
     , SOLD_CUST_GRP_CD
     , SOLD_CUST_GRP_CD AS SHIP_CUST_GRP_CD
     , ITEM_CD
     , SALES.YYMMDD
     , CONCAT(SALES.YYMMDD, '-', FORMAT(ROW_NUMBER() OVER (PARTITION BY SALES.YYMMDD ORDER BY SALES.YYMMDD), '000000')) AS SEQ
     , 0 AS DISCOUNT
     , WEEK
     , PART_WEEK
     , YYMM
     , YY
     , NULL AS USER_CD
     , RST_SALES_QTY
     , RST_SALES_PRICE
     , 'SYSTEM' AS CREATE_USER_CD
--      , GETDATE() AS CREATE_DATE
     , NULL AS MODIFY_USER_CD
     , NULL AS MODIFY_DATE
  FROM (
        SELECT CASE WHEN CHAIN_NM = '이마트' THEN '1065'
                    WHEN CHAIN_NM = '롯데마트' THEN '1066'
                    WHEN CHAIN_NM = '홈플러스' THEN '1067'
                    ELSE '-'
                END AS SOLD_CUST_GRP_CD
             , SELL_BARCD AS ITEM_CD
             , SELL_DATE  AS YYMMDD
             , SELL_QTY   AS RST_SALES_QTY
             , SELL_PRICE AS RST_SALES_PRICE
          FROM TEST.dbo.SELLOUT_PAST_TEMP SALES
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