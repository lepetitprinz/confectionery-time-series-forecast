INSERT INTO M4S_O110600
(
   PROJECT_CD
 , DATA_VRSN_CD
 , DIVISION_CD
 , STAT_CD
 , WEEK
 , FKEY
 , YYMMDD
 , TIME_INDEX
 , RESULT_SALES
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
 , CREATE_USER_CD
 , CREATE_DATE
)
         SELECT SCORE.PROJECT_CD
              , SCORE.DATA_VRSN_CD
              , SCORE.DIVISION_CD
              , SCORE.STAT_CD
              , WEEK
              , SCORE.FKEY
              , PRED.YYMMDD
              , TIME_INDEX
              , PRED.RESULT_SALES
              , PRED.CUST_GRP_CD
              , PRED.CUST_GRP_NM
              , PRED.CUST_CD
              , PRED.CUST_NM
              , PRED.ITEM_ATTR01_CD
              , PRED.ITEM_ATTR01_NM
              , PRED.ITEM_ATTR02_CD
              , PRED.ITEM_ATTR02_NM
              , PRED.ITEM_ATTR03_CD
              , PRED.ITEM_ATTR03_NM
              , PRED.ITEM_ATTR04_CD
              , PRED.ITEM_ATTR04_NM
              , PRED.ITEM_CD
              , PRED.ITEM_NM
              , 'SYSTEM'  AS CREATE_USER_CD
              , GETDATE() AS CREATE_DATE
         FROM M4S_O110610 SCORE
                  INNER JOIN (
             SELECT *
             FROM M4S_I110400
             --where DATA_VRSN_CD = '20210416-20210912'
         ) PRED
                             ON SCORE.PROJECT_CD = PRED.PROJECT_CD
                                 AND SCORE.DATA_VRSN_CD = PRED.DATA_VRSN_CD
                                 AND SCORE.DIVISION_CD = PRED.DIVISION_CD
                                 AND SCORE.FKEY = PRED.FKEY
								 AND SCORE.STAT_CD = PRED.STAT_CD