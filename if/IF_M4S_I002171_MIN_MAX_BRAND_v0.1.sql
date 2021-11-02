declare @START_DAY INT;
declare @END_DAY INT;
set @START_DAY = (SELECT OPTION_VAL FROM M4S_I001020 WHERE OPTION_CD = 'RST_START_DAY');
set @END_DAY = (SELECT OPTION_VAL FROM M4S_I001020 WHERE OPTION_CD = 'RST_END_DAY');

INSERT INTO M4S_I002171
(
   PROJECT_CD
 , DATA_VRSN_CD
 , DIVISION_CD
 , BIZ_CD
 , LINE_CD
 , BRAND_CD
 , MIN_QTY
 , MAX_QTY
 , CREATE_USER_CD
 , CREATE_DATE
)
SELECT PROJECT_CD
     , CONCAT(@START_DAY, '-', @END_DAY) AS DATA_VRSN_CD
     , 'BRAND_CD' AS DIVISION_CD
     , BIZ_CD
     , LINE_CD
     , BRAND_CD
     , MIN(QTY) AS MIN_QTY
     , MAX(QTY) AS MAX_QTY
     , 'SYSTEM' AS CREATE_USER_CD
     , GETDATE() AS CREATE_DATE
  FROM (
        SELECT PROJECT_CD
             , BIZ_CD
             , LINE_CD
             , BRAND_CD
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
                             , RST_SALES_QTY    AS QTY
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
 GROUP BY PROJECT_CD
        , BIZ_CD
        , LINE_CD
        , BRAND_CD