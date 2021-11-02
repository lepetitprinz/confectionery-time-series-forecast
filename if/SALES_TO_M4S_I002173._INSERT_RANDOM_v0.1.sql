INSERT INTO M4S_I002173(
    PROJECT_CD,
    DIVISION_CD,
    SOLD_CUST_GRP_CD,
    SHIP_CUST_GRP_CD,
    ITEM_CD,
    YYMMDD,
    SEQ,
    DISCOUNT,
    WEEK,
    PART_WEEK,
    YYMM,
    YY,
    USER_CD,
    RST_SALES_QTY,
    RST_SALES_PRICE,
    CREATE_USER_CD,
    CREATE_DATE,
    MODIFY_USER_CD,
    MODIFY_DATE
)
SELECT 'ENT001'                                                                                  AS PROJECT_CD
      , 'SELL-OUT'                                                                                AS DIVISION_CD
      , 'LOTTEMART'                                                                               AS SOLD_CUST_GRPP_CD
      , 'LOTTEMART'                                                                               AS SHIP_CUST_GRPP_CD
      , ISNULL(ITEM.ITEM_CD, '')                                                                  AS ITEM_CD
      , CAL.YYMMDD
      , CONCAT(CAL.YYMMDD, '-',
               FORMAT(ROW_NUMBER() OVER (PARTITION BY CAL.YYMMDD ORDER BY CAL.YYMMDD), '000000')) AS SEQ
      , 0                                                                                         AS DISCOUNT
      , CAL.WEEK
      , CAL.PART_WEEK
      , CAL.YYMM
      , CAL.YY
      , 'SYSTEM'                                                                                  AS USER_CD
      , RST_SALES_QTY
      , RST_SALES_PRICE
      , 'SYSTEM' AS CREAT_USER_CD
      , GETDATE() AS CREATE_DATE
      , NULL AS MODIFY_USER_CD
      , NULL AS MODIFY_DATE
  FROM (
        SELECT SOLD_CUST_GRP_CD
             , ITEM_CD
             , DATEADD(MONTH, 4, CONVERT(DATETIME, YYMMDD)) AS YYMMDD
             , RST_SALES_QTY + CONVERT(INT, RAND() * (10 - 1)) AS RST_SALES_QTY
             , RST_SALES_PRICE
          FROM M4S_I002173_TMP
           ) SELL_OUT
  LEFT OUTER JOIN M4S_I002030 CAL
                  ON SELL_OUT.YYMMDD = CAL.YYMMDD
  LEFT OUTER JOIN M4S_I002040 ITEM
                  ON SELL_OUT.ITEM_CD = ITEM.ITEM_ATTR09_CD