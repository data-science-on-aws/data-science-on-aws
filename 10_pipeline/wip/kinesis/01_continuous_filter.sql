-- ** Continuous Filter ** 
-- Performs a continuous filter based on a WHERE condition.
--          .----------.   .----------.   .----------.              
--          |  SOURCE  |   |  INSERT  |   |  DESTIN. |              
-- Source-->|  STREAM  |-->| & SELECT |-->|  STREAM  |-->Destination
--          |          |   |  (PUMP)  |   |          |              
--          '----------'   '----------'   '----------'               
-- STREAM (in-application): a continuously updated entity that you can SELECT from and INSERT into like a TABLE
-- PUMP: an entity used to continuously 'SELECT ... FROM' a source STREAM, and INSERT SQL results into an output STREAM
-- Create output stream, which can be used to send to a destination
CREATE OR REPLACE STREAM "DESTINATION_SQL_STREAM" (ticker_symbol VARCHAR(4), sector VARCHAR(12), change REAL, price REAL);
-- Create pump to insert into output 
CREATE OR REPLACE PUMP "STREAM_PUMP" AS INSERT INTO "DESTINATION_SQL_STREAM"
-- Select all columns from source stream
SELECT STREAM ticker_symbol, sector, change, price
FROM "SOURCE_SQL_STREAM_001"
-- LIKE compares a string to a string pattern (_ matches all char, % matches substring)
-- SIMILAR TO compares string to a regex, may use ESCAPE
WHERE sector SIMILAR TO '%TECH%';

