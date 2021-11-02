SELECT PROJECT_CD
     , IF_VRSN_ID
     , MST.CUST_CD
     , CUST_TYPE_CD
     , CUST_NM
     , SP1.SP1_CD AS CUST_GRP_CD
     , SITE_CD
     , CUST_DIVISION_CD
     , DOME_CD
     , CURCY_CD
     , ADDR
     , ZIP_CD
     , TEL_NO
     , DEL_YN
     , USE_YN
     , DESCR
     , ATTR01_VAL
     , ATTR02_VAL
     , ATTR03_VAL
     , ATTR04_VAL
     , ATTR05_VAL
     , ATTR06_VAL
     , ATTR07_VAL
     , ATTR08_VAL
     , ATTR09_VAL
     , ATTR10_VAL
     , 'matrix' AS CREATE_USER_CD
     , CREATE_DATE
     , MODIFY_USER_CD
     , MODIFY_DATE
  FROM M4S_I002060_BAK MST
 RIGHT OUTER JOIN (
                   SELECT CUST_CD
                        , SP1_CD
                     FROM (
                           SELECT CUST_CD
                                , ISNULL(SP1_CD, '-') AS SP1_CD
                             FROM (
                                   SELECT CUST_CD
                                        , SP1_NM
                                     FROM CUST_SP1_MAP_TMP
                                    GROUP BY CUST_CD
                                           , SP1_NM
                                  ) CUST
                             LEFT OUTER JOIN (
                                              SELECT LINK_SALES_MGMT_CD AS SP1_CD
                                                   , LINK_SALES_MGMT_NM AS SP1_NM
                                               FROM M4S_I204020
                                              WHERE SALES_MGMT_VRSN_ID LIKE '%V1%'
                                            ) SP1
                               ON CUST.SP1_NM = SP1.SP1_NM
                          ) SP1
                    GROUP BY CUST_CD
                           , SP1_CD
                  ) SP1
    ON MST.CUST_CD = SP1.CUST_CD