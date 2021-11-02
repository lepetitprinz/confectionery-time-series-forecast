INSERT INTO M4S_I002170 (
    PROJECT_CD
  , DIVISION_CD
  , SOLD_CUST_GRP_CD
  , SHIP_CUST_GRP_CD
  , ITEM_CD
  , YYMMDD
  , SEQ
  , FROM_DC_CD
  , UNIT_PRICE
  , UNIT_CD
  , DISCOUNT
  , WEEK
  , PART_WEEK
  , YYMM
  , YY
  , USER_CD
  , RST_SALES_QTY
  , RST_SALES_PRICE
  , CREATE_USER_CD
  , CREATE_DATE
  , MODIFY_USER_CD
  , MODIFY_DATE
)
SELECT PROJECT_CD
     , DIVISION_CD
     , SOLD_CUST_GRP_CD
     , SOLD_CUST_GRP_CD AS SHIP_CUST_GRP_CD
     , ITEM_CD
     , YYMMDD
     , CONCAT(YYMMDD, '-', FORMAT(ROW_NUMBER() OVER (PARTITION BY YYMMDD ORDER BY YYMMDD),'000000')) AS SEQ
     , FROM_DC_CD
     , UNIT_PRICE
     , UNIT_CD
     , ISNULL(ROUND(1 - (ABS(ORG_COST) / ABS(FAC_PRICE)), 3), 0) AS DISCOUNT
     , WEEK
     , PART_WEEK
     , YYMM
     , YY
     , USER_CD
     , RST_SALES_QTY
     , RST_SALES_PRICE
     , 'SYSTEM'  AS CREATE_USER_CD
     , getdate() AS CREATE_DATE
     , NULL      AS MODIFY_USER_CD
     , NULL      AS MODIFY_DATE
  FROM (
         SELECT TEMP.PROJECT_CD
              , DIVISION_CD
              , SOLD_CUST_GRP_CD
              , TEMP.ITEM_CD
              , TEMP.YYMMDD
              , FROM_DC_CD
              , UNIT_PRICE
              , TEMP.UNIT_CD
              , CAL.WEEK
              , CAL.PART_WEEK
              , CAL.YYMM
              , CAL.YY
              , ROUND(ORG_COST / RST_SALES_QTY, 2) AS ORG_COST
              , PRICE.FAC_PRICE                    AS FAC_PRICE
              , USER_CD
              , RST_SALES_QTY
              , RST_SALES_PRICE
         FROM (
                  SELECT PROJECT_CD
                       , DIVISION_CD
                       , SOLD_CUST_GRP_CD
                       , ITEM_CD
                       , YYMMDD
                       , FROM_DC_CD
                       , CONVERT(INT, UNIT_PRICE)                 AS UNIT_PRICE
                       , UNIT_CD
                       , CONVERT(NUMERIC(18, 1), ORG_COST)        AS ORG_COST
                       , USER_CD
                       , RST_SALES_QTY * SIGN                     AS RST_SALES_QTY
                       , CONVERT(NUMERIC(18, 0), RST_SALES_PRICE) AS RST_SALES_PRICE
                  FROM (
                           SELECT 'ENT001'                                      AS PROJECT_CD
								, 'SELL_IN'                                     AS DIVISION_CD
								, CONVERT(nvarchar, CONVERT(INT, SALES3))       AS SOLD_CUST_GRP_CD
								, CONVERT(nvarchar, CONVERT(INT, SALES8))       AS ITEM_CD
								, REPLACE(SALES2, '-', '')                      AS YYMMDD
								, SALES31                                       AS FROM_DC_CD
								, REPLACE(REPLACE(SALES15, '.00', ''), '-', '') AS UNIT_PRICE
								, SALES14                                       AS UNIT_CD
								, REPLACE(SALES16, '-', '')                     AS ORG_COST
								, 'SYSTEM'                                      AS USER_CD
								, CONVERT(NUMERIC(18, 1), SALES13)              AS RST_SALES_QTY
								, IIF(RIGHT(SALES19, 1) = '-', -1, 1)           AS SIGN
								, REPLACE(SALES19, '-', '')                     AS RST_SALES_PRICE
                             FROM M4S_I002170_TMP
                       ) TEMP
                  WHERE 1 = 1
                    AND RST_SALES_QTY <> 0 -- REMOVE 0 AMOUNT
              ) TEMP
                  LEFT OUTER JOIN M4S_I002041 PRICE
                    ON TEMP.PROJECT_CD = PRICE.PROJECT_CD
                   AND TEMP.ITEM_CD = PRICE.ITEM_CD
                   AND TEMP.UNIT_CD = PRICE.PRICE_QTY_UNIT_CD
                  LEFT OUTER JOIN M4S_I002030 CAL
                    ON TEMP.PROJECT_CD = CAL.PROJECT_CD
                   AND TEMP.YYMMDD = CAL.YYMMDD
     ) RESULT