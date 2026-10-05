CREATE TABLE IF NOT EXISTS energy_consumption (
    id BIGSERIAL PRIMARY KEY,

    timestamp TIMESTAMP NOT NULL,
    household_id VARCHAR(20) NOT NULL,

    consumption_kwh DOUBLE PRECISION NOT NULL,
    solar_generation_kwh DOUBLE PRECISION NOT NULL,

    temperature_c DOUBLE PRECISION,
    humidity_percent DOUBLE PRECISION,

    hour INTEGER,
    day_of_week INTEGER,
    month INTEGER,
    is_weekend BOOLEAN,

    previous_consumption DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_energy_timestamp
ON energy_consumption(timestamp);

CREATE INDEX IF NOT EXISTS idx_energy_household
ON energy_consumption(household_id);