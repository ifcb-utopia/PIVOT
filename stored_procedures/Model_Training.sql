CREATE OR ALTER PROCEDURE AL_TRAIN_SET
    @MODEL_ID INT,
    @D_METRIC_ID INT,
    @TRAIN_SIZE INT,
    @IMAGE_IDS VARCHAR(MAX) -- other image_ids to be excluded from sampling

AS
BEGIN
    DECLARE @EXCLUDE_IDS TABLE (I_ID INT);

    -- Convert comma-separated string to a table variable
    INSERT INTO @EXCLUDE_IDS (I_ID)
    SELECT CAST(value AS INT)
    FROM STRING_SPLIT(@IMAGE_IDS, ',');

    WITH LABEL_COUNTS AS (
        SELECT I_ID,
               LABEL,
               SUM(WEIGHT) AS W_COUNT,
               SUM(WEIGHT) AS W_COUNT,
--                SUM(SUM(WEIGHT)) OVER (PARTITION BY I_ID) AS TOTAL_SUM,
               CAST(SUM(WEIGHT) AS FLOAT) / (SUM(SUM(WEIGHT)) OVER (PARTITION BY I_ID)) as PERCENT_CONSENSUS
        FROM LABELS
        GROUP BY I_ID, LABEL
    ),
    LABEL_STATS AS (
        SELECT I_ID,
               STRING_AGG(LABEL, ', ') WITHIN GROUP (ORDER BY W_COUNT DESC) AS ALL_LABELS,
               STRING_AGG(CAST(PERCENT_CONSENSUS AS NVARCHAR(255)), ', ') WITHIN GROUP (ORDER BY W_COUNT DESC) AS LABEL_PERCENTS
        FROM LABEL_COUNTS
        GROUP BY I_ID
    ),
    TEST_IMAGES AS (
        SELECT DISTINCT I_ID
        FROM METRICS
        WHERE D_ID = 0
    )
    SELECT TOP (@TRAIN_SIZE)
           I.I_ID AS IMAGE_ID,
           I.FILEPATH AS BLOB_FILEPATH,
           L.ALL_LABELS AS ALL_LABELS,
           L.LABEL_PERCENTS AS LABEL_PERCENTS,
           M.D_VALUE AS UNCERTAINTY
    FROM METRICS AS M
    INNER JOIN IMAGES AS I
        ON M.I_ID = I.I_ID
    INNER JOIN PREDICTIONS AS P
        ON M.I_ID = P.I_ID AND M.M_ID = P.M_ID
    INNER JOIN LABEL_STATS AS L
        ON M.I_ID = L.I_ID
    LEFT JOIN TEST_IMAGES AS TI
        ON I.I_ID = TI.I_ID
    LEFT JOIN @EXCLUDE_IDS O_EI
        ON I.I_ID = O_EI.I_ID
    WHERE TI.I_ID IS NULL
      AND O_EI.I_ID IS NULL
      AND M.M_ID = @MODEL_ID
      AND M.D_ID = @D_METRIC_ID
    ORDER BY UNCERTAINTY DESC;
END;