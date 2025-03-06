SHOW Databases
USE traveltrip

CREATE TABLE passenger (
    usercode INT PRIMARY KEY ,  
    company VARCHAR(255),                    
    name VARCHAR(255),                       
    gender ENUM('Male', 'Female', 'Other')   
);

DROP TABLE passenger;


CREATE TABLE flight (
    travelcode INT PRIMARY KEY, 
    user_id INT,
    departure VARCHAR(100),
    arrival VARCHAR(100),
    flight_type VARCHAR(50),
    flight_price DECIMAL(10, 2),
    flight_duration INT, 
    flight_distance INT, 
    flight_agency VARCHAR(100),
    departure_date DATETIME,
    FOREIGN KEY (user_id) REFERENCES passenger(usercode)
);


SELECT * from flight;
DESCRIBE flight;
USE traveltrip;







DESCRIBE passenger;
DESCRIBE user;
USE traveltrip;
SELECT * from passenger
DESCRIBE flight;
USE traveltrip;
DESCRIBE hotel;

USE traveltrip;
DESCRIBE flight;
CREATE TABLE hotel (
    user_id INT,
    travel_code INT ,
    hotel_name VARCHAR(100),
    arrival_place VARCHAR(100),
    hotel_stay INT, 
    hotel_per_day_rent DECIMAL(10, 2),
    check_in DATETIME,
    hotel_total_price DECIMAL(10, 2),
    FOREIGN KEY (travel_code) REFERENCES flight(travelcode)
);
DESCRIBE hotel;

DESCRIBE passenger;
SELECT * FROM flight;

USE traveltrip;

SELECT * FROM user WHERE user_id IN (SELECT DISTINCT user_id FROM flight);


ALTER TABLE passenger MODIFY gender VARCHAR(10);
-- USE traveltrip
-- ALTER TABLE hotel DROP FOREIGN KEY flight_ibfk_2;

-- USE traveltrip;
-- DROP TABLE flight;

-- SELECT CONSTRAINT_NAME
-- FROM information_schema.KEY_COLUMN_USAGE
-- WHERE TABLE_NAME = 'hotel'
-- AND TABLE_SCHEMA = 'traveltrip';

-- ALTER TABLE hotel DROP FOREIGN KEY hotel_ibfk_1;

-- SELECT DATABASE();

-- USE traveltrip;
-- ALTER TABLE hotel DROP FOREIGN KEY hotel_ibfk_1;

-- DESCRIBE hotel;
-- USE traveltrip;
-- ALTER TABLE hotel DROP FOREIGN KEY flight_ibfk_1;
-- FOREIGN KEY (user_id) REFERENCES user(user_id);

-- DESC user;


-- USE traveltrip;
-- DESC hotel;
-- DESC flight;

-- SELECT * FROM hotel ;

-- USE traveltrip;
-- DROP TABLE hotel;


USE traveltrip;
CREATE TABLE guest_profile (
    Guest_Id INT PRIMARY KEY,
    TravelCode INT,
    Guest_Name VARCHAR(100),
    Guest_Gender VARCHAR(10),
    Age INT,
    Guest_PhoneNo VARCHAR(15),
    Guest_Email VARCHAR(100),
    IdProof VARCHAR(50),
    FOREIGN KEY (TravelCode) REFERENCES flight(TravelCode)
);

DESCRIBE guest_profile
ALTER TABLE guest_profile MODIFY Guest_PhoneNo VARCHAR(50);
ALTER TABLE guest_profile MODIFY Guest_Id INT;
USE traveltrip;
ALTER TABLE guest_profile DROP PRIMARY KEY;
USE traveltrip;
CREATE TABLE car_rent (
    Rent_ID INT AUTO_INCREMENT PRIMARY KEY,
    User_ID INT,  -- Reference to the user table
    TravelCode INT,  -- Reference to the flights table
    Rent_Date DATETIME,
    Pickup_Location VARCHAR(255) NOT NULL,
    Dropoff_Location VARCHAR(255) NOT NULL,
    Car_Type VARCHAR(100) NOT NULL,
    Rental_Agency VARCHAR(100) NOT NULL,
    Rental_Duration INT NOT NULL,  -- Duration in days
    Car_Total_Distance DECIMAL(10,2) NOT NULL,  -- Distance in km
    Fuel_Policy VARCHAR(50) NOT NULL,  -- E.g., Full-to-Full, Prepaid, etc.
    Car_BookingStatus VARCHAR(50) NOT NULL,  -- E.g., Confirmed, Pending, Cancelled
    Total_Rent_Price DECIMAL(10,2) NOT NULL,  -- Total rental cost

    -- Foreign Key Constraints
    FOREIGN KEY (User_ID) REFERENCES passenger(usercode),
    FOREIGN KEY (TravelCode) REFERENCES flight(travelcode)
);

DESCRIBE car_rent;
USE traveltrip;
SELECT * FROM car_rent;



SELECT CONSTRAINT_NAME
FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_NAME = 'flight'
AND TABLE_SCHEMA = 'traveltrip';

USE traveltrip;
DESCRIBE flight;


DROP TABLE IF EXISTS flight;

-- CREATE TABLE flight (
--     travelcode VARCHAR(50) PRIMARY KEY,  
--     usercode INT,  -- Reference to passenger
--     departure VARCHAR(255) NOT NULL,
--     arrival VARCHAR(255) NOT NULL,
--     flight_type VARCHAR(50) NOT NULL,
--     flight_price DECIMAL(10,2) NOT NULL,
--     flight_duration TIME NOT NULL,
--     flight_distance DECIMAL(10,2) NOT NULL,
--     flight_agency VARCHAR(255) NOT NULL,
--     departure_date DATE NOT NULL,

--     -- Foreign Key Constraints
--     CONSTRAINT fk_flight_passenger FOREIGN KEY (usercode) REFERENCES passenger(usercode) ON DELETE CASCADE ON UPDATE CASCADE
-- );

DROP TABLE car_rent;
USE traveltrip;
SELECT * from flight;
SELECT * from passenger;
CREATE TABLE review (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    travelCode INT,  
    User_ID INT,
    car_rented BOOLEAN DEFAULT FALSE,
    review_car TEXT,
    review_hotel TEXT,
    review_flights TEXT,
    flight_rating INT CHECK (flight_rating BETWEEN 1 AND 5),
    hotel_rating INT CHECK (hotel_rating BETWEEN 1 AND 5),
    car_rating INT CHECK (car_rating BETWEEN 1 AND 5),
    overall_rating DECIMAL(3,2) CHECK (overall_rating BETWEEN 1 AND 5),
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (travelCode) REFERENCES flight(travelCode) ON DELETE CASCADE,
    FOREIGN KEY (User_ID) REFERENCES passenger(usercode) ON DELETE CASCADE
);
ALTER TABLE review MODIFY review_car VARCHAR(500);

ALTER TABLE review MODIFY review_flights VARCHAR(500);
ALTER TABLE review DROP CONSTRAINT review_chk_3;


ALTER table review DROP review_date;
ALTER table review DROP car_rented;
ALTER table review MODIFY car_rating INT;
ALTER table review MODIFY flight_rating INT;
ALTER table review MODIFY hotel_rating INT;
ALTER table review MODIFY overall_rating INT;
ALTER TABLE review ADD CONSTRAINT overall_rating CHECK (overall_rating BETWEEN 0 AND 10);
ALTER TABLE review MODIFY overall_rating DECIMAL(3,2);

SELECT * FROM car_rent;

DESCRIBE review

USE traveltrip;
CREATE TABLE Merged (
    User_ID INT,
    company VARCHAR(20),
    name VARCHAR(20),
    User_gender VARCHAR(10),
    User_age INT,
    travelcode INT PRIMARY KEY,
    departure_place VARCHAR(50),
    arrival_place VARCHAR(50),
    flight_type VARCHAR(10),
    flight_price DECIMAL(10,2),
    flight_duration TIME,
    flight_distance DECIMAL(10,2),
    flight_agency VARCHAR(20),
    departure_date DATETIME,
    hotel_name VARCHAR(15),
    hotel_stay INT,
    hotel_price_per_day DECIMAL(10,2),
    hotel_total_price DECIMAL(10,2),
    check_in DATETIME,
    Pickup_Location VARCHAR(15),
    Dropoff_Location VARCHAR(15),
    car_type VARCHAR(10),
    car_agency VARCHAR(15),
    car_rental_duration INT,
    Car_Total_Distance DECIMAL(10,2),
    Fuel_Policy VARCHAR(10),
    Car_BookingStatus VARCHAR(15),
    car_total_price DECIMAL(10,2),
    total_trip_cost DECIMAL(10,2)
    );
DESCRIBE Merged;
SELECT * from Merged

USE traveltrip;
ALTER TABLE Merged MODIFY name VARCHAR(60);
ALTER TABLE Merged MODIFY Fuel_Policy VARCHAR(60);

ALTER TABLE Merged MODIFY Pickup_Location VARCHAR(60);
ALTER TABLE Merged MODIFY Dropoff_Location VARCHAR(60);

ALTER TABLE Merged DROP PRIMARY KEY;

DROP TABLE Merged

CREATE TABLE Test_Car (
    Rent_ID INT AUTO_INCREMENT PRIMARY KEY,
    User_ID INT,  -- Reference to the user table
    TravelCode INT,  -- Reference to the flights table
    Rent_Date DATETIME,
    Pickup_Location VARCHAR(255) NOT NULL,
    Dropoff_Location VARCHAR(255) NOT NULL,
    Car_Type VARCHAR(100) NOT NULL,
    Rental_Agency VARCHAR(100) NOT NULL,
    Rental_Duration INT NOT NULL,  -- Duration in days
    Car_Total_Distance DECIMAL(10,2) NOT NULL,  -- Distance in km
    Fuel_Policy VARCHAR(50) NOT NULL,  -- E.g., Full-to-Full, Prepaid, etc.
    Car_BookingStatus VARCHAR(50) NOT NULL  -- E.g., Confirmed, Pending, Cancelled
)

desc Test_Car

select * from car_rent;
Use traveltrip;
SELECT * from Test_car;
