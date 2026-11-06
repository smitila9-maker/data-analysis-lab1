-- public.aircraft_types определение

-- Drop table

-- DROP TABLE public.aircraft_types;

CREATE TABLE public.aircraft_types (
	type_code varchar(20) NOT NULL,
	description varchar(128) NULL,
	CONSTRAINT aircraft_types_pkey PRIMARY KEY (type_code),
	CONSTRAINT aircraft_types_type_code_not_null NOT NULL type_code
);


-- public.airports определение

-- Drop table

-- DROP TABLE public.airports;

CREATE TABLE public.airports (
	airport_code varchar(10) NOT NULL,
	airport_name varchar(128) NOT NULL,
	CONSTRAINT airports_airport_code_not_null NOT NULL airport_code,
	CONSTRAINT airports_airport_name_not_null NOT NULL airport_name,
	CONSTRAINT airports_pkey PRIMARY KEY (airport_code)
);


-- public.flights определение

-- Drop table

-- DROP TABLE public.flights;

CREATE TABLE public.flights (
	id serial4 NOT NULL,
	callsign varchar(20) NULL,
	aircraft_uid varchar(50) NULL,
	typecode varchar(20) NULL,
	origin varchar(10) NULL,
	destination varchar(10) NULL,
	firstseen timestamp NULL,
	lastseen timestamp NULL,
	"day" date NULL,
	latitude_1 float8 NULL,
	longitude_1 float8 NULL,
	altitude_1 float8 NULL,
	latitude_2 float8 NULL,
	longitude_2 float8 NULL,
	altitude_2 float8 NULL,
	CONSTRAINT flights_id_not_null NOT NULL id,
	CONSTRAINT flights_pkey PRIMARY KEY (id),
	CONSTRAINT flights_destination_fkey FOREIGN KEY (destination) REFERENCES public.airports(airport_code),
	CONSTRAINT flights_origin_fkey FOREIGN KEY (origin) REFERENCES public.airports(airport_code),
	CONSTRAINT flights_typecode_fkey FOREIGN KEY (typecode) REFERENCES public.aircraft_types(type_code)
);