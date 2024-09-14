/*
Name: METRICS_SELECTION
Description: This stored procedure retrieves a subsection of the metrics table given a d_id and container.
Parameters:
- @MODEL_ID: Integer indicating model ID
- @D_METRIC_ID: Integer denoting dissimilarity metric ID
- @CONTAINER: Indicates which container is active
*/

CREATE OR ALTER PROCEDURE METRICS_SELECTION
    @MODEL_ID INT,
    @D_METRIC_ID INT,
    @CONTAINER VARCHAR(255)
AS
BEGIN
	SELECT
		images.i_id AS I_ID,
		metrics.m_id AS M_ID,
		metrics.d_id AS D_ID,
		metrics.d_value AS D_VALUE,
		images.container AS CONTAINER
	FROM metrics
	INNER JOIN images
		ON metrics.i_id = images.i_id
	WHERE metrics.m_id = @MODEL_ID
		AND metrics.d_id = @D_METRIC_ID
		AND images.container = @CONTAINER;
END;