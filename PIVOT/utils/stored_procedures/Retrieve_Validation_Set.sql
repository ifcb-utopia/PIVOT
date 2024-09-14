/*
Name: VALIDATION_SET
Description: This stored procedure retrieves a set of specified images.
Parameters:
- @I_ID_LIST: A list of image ids to retrieve information about.
- @CONTAINER: Indicates which container is active
- @MODEL_ID: Model ID number
- @DISS_ID: Dissimilarity ID number
*/

CREATE OR ALTER PROCEDURE VALIDATION_SET
    @I_ID_LIST IDList READONLY,
    @CONTAINER VARCHAR(255),
    @MODEL_ID INT,
    @DISS_ID INT
AS
BEGIN
    SELECT 
        I.I_ID AS IMAGE_ID,
        I.FILEPATH AS BLOB_FILEPATH,
        M.D_VALUE AS UNCERTAINTY,
        P.PRED_LABEL AS PRED_LABEL,
        P.CLASS_PROB AS PROBS,
        M.D_VALUE AS RANK_SCORE
    FROM METRICS AS M
    INNER JOIN IMAGES AS I 
        ON M.I_ID = I.I_ID
    INNER JOIN PREDICTIONS AS P
        ON M.I_ID = P.I_ID AND (M.M_ID = P.M_ID OR M.M_ID = 0)
    WHERE
        I.CONTAINER = @CONTAINER
        AND I.I_ID IN (SELECT * FROM @I_ID_LIST)
        AND P.M_ID = @MODEL_ID
        AND M.D_ID = @DISS_ID
END;