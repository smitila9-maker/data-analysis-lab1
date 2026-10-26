-- aircraft_types определение

CREATE TABLE aircraft_types (
    type_code VARCHAR(20) PRIMARY KEY,
    description VARCHAR(128)
);


-- airports определение

CREATE TABLE airports (
    airport_code VARCHAR(10) PRIMARY KEY,
    airport_name VARCHAR(128) NOT NULL
);


-- flights определение

CREATE TABLE flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    callsign VARCHAR(20),
    aircraft_uid VARCHAR(50),
    typecode VARCHAR(20) REFERENCES aircraft_types(type_code),
    origin VARCHAR(10) REFERENCES airports(airport_code),
    destination VARCHAR(10) REFERENCES airports(airport_code),
    firstseen TIMESTAMP,
    lastseen TIMESTAMP,
    day DATE,
    latitude_1 FLOAT,
    longitude_1 FLOAT,
    altitude_1 FLOAT,
    latitude_2 FLOAT,
    longitude_2 FLOAT,
    altitude_2 FLOAT
);

CREATE INDEX idx_flights_origin ON flights(origin);