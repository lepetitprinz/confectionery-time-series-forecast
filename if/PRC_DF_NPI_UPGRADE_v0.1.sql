create PROCEDURE [dbo].[PRC_DF_NPI_UPGRADE]
(
    -- 출력 매개변수
    @O INT OUTPUT, -- SYS_REFCURSOR,

    @V_PROJECT_CD       VARCHAR(50),
    @V_ITEM_CD          VARCHAR(50),
    @V_RATE             INT
    )
AS
BEGIN

/**********************************************************************************/
/* Project       : M4Plan Suites                                                  */
/* Module        : 수요예측                                                        */
/* Program Name  : PRC_DF_NPI_UPGRADE                                             */
/* Description   : NPI UPGRADE 프로시저                                            */
/* Referenced by :                                                                */
/* Program History                                                                */
/**********************************************************************************/
/* Date             In Charge         Description                                 */
/**********************************************************************************/
/* 2021-09-03       E.J.PARK          Initial Release                             */
/**********************************************************************************/

DECLARE @V_PROC_NM    VARCHAR(50); -- 프로시저이름
DECLARE @V_RATE INT;

SET NOCOUNT ON; -- 동작
    -- 프로시저 이름
    SET @V_PROC_NM = 'PRC_DF_NPI_UPGRADE';
    exec dbo.MTX_SCM_PROC_LOG @V_PROJECT_CD, @V_PROC_NM,
        'PRC_DF_NPI_UPGRADE 프로시저', 'ALL START';

    --01.START--
    exec dbo.MTX_SCM_PROC_LOG @V_PROJECT_CD, @V_PROC_NM,
        '(1) START', '01.START';


----------------------------
-- NPI UPGRADE
----------------------------
SELECT PROJECT_CD
     , DATA_VRSN_CD
     , DIVISION_CD
     , STAT_CD
     , WEEK
     , YYMMDD
     , RESULT_SALES * @V_RATE / 100 AS RESULT_SALES
 FROM M4S_O110600
WHERE 1=1
  AND ITEM_CD = @V_ITEM_CD


        --01.END--
    exec dbo.MTX_SCM_PROC_LOG @V_PROJECT_CD, @V_PROC_NM,
        '(1) END', '01.END';

    DECLARE @result INT
	SET @result = 0 -- 0:성공

	IF @@ERROR != 0 SET @result = @@ERROR
	--SELECT @result

	IF(@result <> 0)
		--RETURN(1); --
		SELECT @O = 1 ;
	else
		--RETURN(2); --
		SELECT @O = 2;

    exec dbo.MTX_SCM_PROC_LOG V_PROJECT_CD, V_PROC_NM,
        SQLERRM, 'ALL END';

END
GO