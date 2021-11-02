declare @START_DAY INT;
declare @END_DAY INT;
set @START_DAY = (SELECT OPTION_VAL FROM M4S_I001020 WHERE OPTION_CD = 'RST_START_DAY');
set @END_DAY = (SELECT OPTION_VAL FROM M4S_I001020 WHERE OPTION_CD = 'RST_END_DAY');

INSERT INTO M4S_I002172
(
  PROJECT_CD
, DATA_VRSN_CD
, DIVISION_CD
, BIZ_CD
, LINE_CD
, BRAND_CD
, YYMMDD
, SEQ
, QTY_SCALING
, CREATE_USER_CD
, CREATE_DATE
)
SELECT SALES.PROJECT_CD
     , MM.DATA_VRSN_CD
     , MM.DIVISION_CD
     , SALES.BIZ_CD
     , SALES.LINE_CD
     , SALES.BRAND_CD
     , YYMMDD
     , CONCAT(YYMMDD, '-', FORMAT(ROW_NUMBER() over (PARTITION BY YYMMDD ORDER BY YYMMDD), '000')) AS SEQ
     , (QTY - MIN_QTY) / (MAX_QTY - MIN_QTY) AS QTY_SCALING
     , 'SYSTEM' AS CREAT_USER_CD
     , GETDATE() AS CREATE_DATE
  FROM (
        SELECT PROJECT_CD
             , BIZ_CD
             , LINE_CD
             , BRAND_CD
             , YYMMDD
             , SUM(QTY) AS QTY
         FROM (
               SELECT SALES.PROJECT_CD AS PROJECT_CD
                    , ITEM_ATTR01_CD   AS BIZ_CD
                    , ITEM_ATTR02_CD   AS LINE_CD
                    , ITEM_ATTR03_CD   AS BRAND_CD
                    , YYMMDD
                    , QTY
                 FROM (
                       SELECT PROJECT_CD
                            , ITEM_CD
                            , YYMMDD
                            , RST_SALES_QTY AS QTY
                         FROM M4S_I002170
                        WHERE 1 = 1
                          AND YYMMDD BETWEEN @START_DAY AND @END_DAY
                       ) SALES
                INNER JOIN (
                            SELECT PROJECT_CD
                                 , ITEM_CD
                                 , ITEM_ATTR01_CD
                                 , ITEM_ATTR02_CD
                                 , ITEM_ATTR03_CD
                              FROM M4S_I002040
                            ) COMM
                   ON SALES.PROJECT_CD = COMM.PROJECT_CD
                  AND SALES.ITEM_CD = COMM.ITEM_CD
              ) SALES
        GROUP BY PROJECT_CD
               , BIZ_CD
               , LINE_CD
               , BRAND_CD
              , YYMMDD
        ) SALES
 LEFT OUTER JOIN (
                  SELECT *
                    FROM M4S_I002171
                   WHERE 1=1
                     AND DIVISION_CD = 'BRAND_CD'
                 ) MM
   ON SALES.PROJECT_CD = MM.PROJECT_CD
  AND SALES.BIZ_CD = MM.BIZ_CD
  AND SALES.LINE_CD = MM.LINE_CD
  AND SALES.BRAND_CD = MM.BRAND_CD