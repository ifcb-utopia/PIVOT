CREATE PROCEDURE AL_RANKINGS
    @MODEL_ID INT,
    @D_METRIC_ID INT,
    @RELABEL_LAMBDA FLOAT,
    @BATCH_SIZE INT
AS
WITH LABEL_COUNTS AS (
    SELECT
        I_ID,
        SUM(WEIGHT) AS W_COUNT
    FROM LABEL
    GROUP BY I_ID
)
SELECT TOP (@BATCH_SIZE)
       I.I_ID AS IMAGE_ID,
       I.FILEPATH AS BLOB_FILEPATH,
       D.D_VALUE AS UNCERTAINTY,
       P.PRED_LABEL AS PRED_LABEL,
       (D.D_VALUE * EXP(-@RELABEL_LAMBDA * LABEL_COUNTS.W_COUNT) * (LABEL_COUNTS.W_COUNT < 10)) AS RANK_SCORE
FROM METRICS AS M
INNER JOIN IMAGES AS I
    ON M.I_ID = I.I_ID
INNER JOIN PREDICTIONS AS P
    ON M.I_ID = P.I_ID AND M.M_ID = P.M_ID
INNER JOIN LABEL_COUNTS AS L
    ON M.I_ID = L.I_ID
WHERE
      M.M_ID = @MODEL_ID
  AND M.D_ID = @D_METRIC_ID
ORDER BY RANK_SCORE DESC
--LIMIT @BATCH_SIZE
GO;