CREATE TABLE IF NOT EXISTS forecast_results (
    id BIGSERIAL PRIMARY KEY,

    household_id VARCHAR(20) NOT NULL,
    input_timestamp TIMESTAMP NOT NULL,

    predicted_consumption_kwh DOUBLE PRECISION NOT NULL,

    model_uri TEXT NOT NULL,
    forecasted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_forecast_household_timestamp
ON forecast_results (household_id, input_timestamp);

CREATE INDEX IF NOT EXISTS idx_forecast_generated_at
ON forecast_results (forecasted_at);