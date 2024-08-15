-- Trying to implement df.describe() for numeric columns
SELECT AVG(`cost`) as average_cost, Avg(`rate`) as average_rating, Avg(`votes`) as average_votes FROM `suggester-423307.cleaned_file.table1` ;

-- Trying to implement df.describe() for string and boolean type columns
SELECT COUNT(DISTINCT `name`) AS unique_values, COUNTIF(`online_order` = TRUE) AS true_count, COUNTIF(`online_order` = FALSE) AS false_count
FROM  `suggester-423307.cleaned_file.table1`;

-- Distribution of online ordering facility provided by Restaurants. 
SELECT  online_order, COUNT(*) AS online_order_count FROM  `suggester-423307.cleaned_file.table1` GROUP BY online_order ORDER BY online_order_count DESC;

--  Ditribution of loacation as to top areas in Bangalore with highest restaurants
WITH filtered_data AS ( SELECT * FROM `suggester-423307.cleaned_file.table1` WHERE location != 'others')
SELECT location, COUNT(*) AS count FROM filtered_data GROUP BY  location ORDER BY count DESC LIMIT 10 ;

-- Distribution of table booking facility
SELECT  book_table,  COUNT(*) AS count FROM  `suggester-423307.cleaned_file.table1` GROUP BY book_table;

-- Types of  Restaurant
SELECT rest_type, COUNT(*) AS count FROM `suggester-423307.cleaned_file.table1`
WHERE rest_type IN (SELECT DISTINCT rest_type FROM `suggester-423307.cleaned_file.table1` LIMIT 10) GROUP BY rest_type ORDER BY count DESC;

-- Top 5 Dishes liked by the users
SELECT dish_liked, COUNT(*) AS dish_liked_count FROM `suggester-423307.cleaned_file.table1` GROUP BY dish_liked ORDER BY dish_liked_count DESC LIMIT 5;

-- Top common Cuisines served across the restaurants
WITH filtered_data AS ( SELECT * FROM `suggester-423307.cleaned_file.table1`  WHERE cuisines != 'others')
SELECT  cuisines, COUNT(*) AS cuisine_count FROM filtered_data GROUP BY cuisines LIMIT 10; 

-- Top 10 Categories of restaurants. 
SELECT type, COUNT(*) AS type_count FROM  `suggester-423307.cleaned_file.table1` GROUP BY type ORDER BY type_count DESC;

-- Distribution of Rating
SELECT rate FROM `suggester-423307.cleaned_file.table1` WHERE rate != 0;

-- Distribution of Cost which is applied to dine for two people

-- This table creates the bins for the cost column
WITH cost_bins AS ( SELECT NTILE(10) OVER (ORDER BY cost) AS cost_bin,  cost  FROM `suggester-423307.cleaned_file.table1`)
SELECT cost_bin, COUNT(*) AS count FROM cost_bins GROUP BY cost_bin ORDER BY cost_bin;
--This query calculates the density of each cost value by dividing its count by the total number of records.
WITH cost_distribution AS (SELECT cost, COUNT(*) AS total_count FROM `suggester-423307.cleaned_file.table1` GROUP BY cost)
SELECT cost, (COUNT(*) / SUM(total_count)) AS density FROM cost_distribution GROUP BY cost;

-- Relationship between Rating and Online Ordering facility
SELECT
  online_order,
  MIN(rate) AS min_rate,
  PERCENTILE_CONT(0, 0.25) OVER (PARTITION BY online_order) AS q1,
  PERCENTILE_CONT(0.5, 0.5) OVER (PARTITION BY online_order) AS median,
  PERCENTILE_CONT(0.75,1) OVER (PARTITION BY online_order) AS q3,
  MAX(rate) AS max_rate FROM `suggester-423307.cleaned_file.table1`
GROUP BY  online_order;

-- Relationship between rating and online reserveration facility
SELECT
  book_table,
  MIN(rate) AS min_rate,
  PERCENTILE_CONT(0,0.25) OVER (PARTITION BY book_table) AS q1,
  PERCENTILE_CONT(0.5,0.5) OVER (PARTITION BY book_table) AS median,
  PERCENTILE_CONT(0.75,1) OVER (PARTITION BY book_table) AS q3,
  MAX(rate) AS max_rate
FROM `suggester-423307.cleaned_file.table1` GROUP BY book_table;

-- Location wise online ordering facility

SELECT location, online_order, COUNT(*) AS count FROM `suggester-423307.cleaned_file.table1` GROUP BY location, online_order;

--Location wise onilne reservation facility
SELECT location, book_table, COUNT(*) AS count From `suggester-423307.cleaned_file.table1` GROUP BY location, book_table;

--Location wise cost ditribution
SELECT location, AVG(cost) as average_cost FROM `suggester-423307.cleaned_file.table1` GROUP BY Location, cost;

-- Top 7 Types of restaurants
SELECT type, COUNT(*) AS count FROM `suggester-423307.cleaned_file.table1` GROUP BY type ORDER BY count DESC LIMIT 10;

-- Top 10 Cuisines served across all the restaurants
SELECT cuisines, COUNT(*) AS cuisine_count FROM  `suggester-423307.cleaned_file.table1` GROUP BY cuisines ORDER BY cuisine_count DESC LIMIT 10;










