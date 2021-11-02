INSERT INTO M4S_O110610
(
   PROJECT_CD
 , DATA_VRSN_CD
 , DIVISION_CD
 , CUST_GRP_CD
 , CUST_GRP_NM
 , CUST_CD
 , CUST_NM
 , ITEM_ATTR01_CD
 , ITEM_ATTR01_NM
 , ITEM_ATTR02_CD
 , ITEM_ATTR02_NM
 , ITEM_ATTR03_CD
 , ITEM_ATTR03_NM
 , ITEM_ATTR04_CD
 , ITEM_ATTR04_NM
 , ITEM_CD
 , ITEM_NM
 , FKEY
 , STAT_CD
 , RMSE
 , CREATE_USER_CD
 , CREATE_DATE
)
SELECT PROJECT_CD
     , DATA_VRSN_CD
     , DIVISION_CD
     , CUST_GRP_CD
     , CUST_GRP_NM
     , CUST_CD
     , CUST_NM
     , ITEM_ATTR01_CD
     , ITEM_ATTR01_NM
     , ITEM_ATTR02_CD
     , ITEM_ATTR02_NM
     , ITEM_ATTR03_CD
     , ITEM_ATTR03_NM
     , ITEM_ATTR04_CD
     , ITEM_ATTR04_NM
     , ITEM_CD
     , ITEM_NM
     , FKEY
     , STAT_CD
     , RMSE
     , 'SYSTEM' AS CREATE_USER_CD
     , GETDATE() AS CREATE_DATE
  FROM (
        SELECT PROJECT_CD
             , DATA_VRSN_CD
             , DIVISION_CD
             , CUST_GRP_CD
             , CUST_GRP_NM
             , CUST_CD
             , CUST_NM
             , ITEM_ATTR01_CD
             , ITEM_ATTR01_NM
             , ITEM_ATTR02_CD
             , ITEM_ATTR02_NM
             , ITEM_ATTR03_CD
             , ITEM_ATTR03_NM
             , ITEM_ATTR04_CD
             , ITEM_ATTR04_NM
             , ITEM_CD
             , ITEM_NM
             , FKEY
             , STAT_CD
             , ROW_NUMBER() over (PARTITION BY PROJECT_CD, DATA_VRSN_CD, DIVISION_CD, CUST_GRP_CD,
                                               ITEM_ATTR01_CD, ITEM_ATTR02_CD, ITEM_ATTR03_CD, ITEM_ATTR04_CD,
                                               ITEM_CD ORDER BY RMSE) AS RANK
             , RMSE
          FROM (
                   SELECT *
                   FROM M4S_I110410
                   WHERE 1 = 1
                     and DATA_VRSN_CD = '20210416-20210912'
                     and fkey like '%C1%'
               ) SCORE
       ) SCORE
WHERE RANK = 1