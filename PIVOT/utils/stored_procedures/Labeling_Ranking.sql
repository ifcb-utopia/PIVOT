/*
Name: AL_RANKINGS
Description: This stored procedure retrieves top-ranked images based on a specified metric and model ID.
Parameters:
- @MODEL_ID: Integer denoting Model ID for filtering predictions.
- @D_METRIC_ID: Integer denoting AL metric ID.
- @BATCH_SIZE: Integer denoting number of top-ranked images to retrieve.
- @CONTAINER: String denoting container name for filtering images.
*/

CREATE OR ALTER PROCEDURE AL_RANKINGS
    @MODEL_ID INT,
    @D_METRIC_ID INT,
    @BATCH_SIZE INT,
	@CONTAINER VARCHAR(255)
AS
BEGIN
    SELECT TOP (@BATCH_SIZE)
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
        ON M.I_ID = P.I_ID AND (M.M_ID = P.M_ID OR M.M_ID = 0) -- allow for test images that have M_ID=0
    WHERE
          P.M_ID = @MODEL_ID
      AND M.D_ID = @D_METRIC_ID
	  AND I.CONTAINER = @CONTAINER
    ORDER BY RANK_SCORE DESC;
END;