--CATEGORICAL COLUMNS Relationship Analysis

-- Question 1
--Is there a relationship between the type of restaurant (rest_type) and the availability of online ordering (online_order)?

-- Explanation
/*
- Type of the restaurant is a categorical column and Online order is a boolean Column. The best test that can be performed is a Chi Square test of independence

CHI SQUARE TEST

I have opted the following procedure to perform the CHI SQUARE TEST

* Create a contingency table: The contingency_table CTE calculates the count of occurrences for each combination of rest_type and online_order.
* Calculate total count: The total_count CTE calculates the total number of rows in the dataset.
* Calculate expected counts: The expected_counts CTE calculates the expected count for each cell using the formula (row_total * column_total) / total_count.
* Calculate Chi-Square statistic: The final SELECT statement calculates the Chi-Square statistic by summing the squared difference between observed and expected counts divided by expected counts.
*/ 
WITH contingency_table AS (
  SELECT rest_type, online_order, COUNT(*) AS count
  FROM `suggester-423307.cleaned_file.table1`
  GROUP BY rest_type, online_order
),
row_totals AS (
  SELECT rest_type, SUM(count) AS row_total
  FROM contingency_table
  GROUP BY rest_type
),
col_totals AS (
  SELECT online_order, SUM(count) AS col_total
  FROM contingency_table
  GROUP BY online_order
),
total_count AS (
  SELECT SUM(count) AS total_count
  FROM contingency_table
),
expected_counts AS (
  SELECT c.rest_type, c.online_order,
    (r.row_total * co.col_total) / t.total_count AS expected_count
  FROM contingency_table c
  JOIN row_totals r ON c.rest_type = r.rest_type
  JOIN col_totals co ON c.online_order = co.online_order
  CROSS JOIN total_count t
)
SELECT SUM(POWER(count - expected_count, 2) / expected_count) AS chi_square_statistic
FROM contingency_table
JOIN expected_counts
USING (rest_type, online_order);

/*
-- Question 2 
--Does the availability of online reservations (book_table) influence the cost category of a restaurant (cost_category)?

Explanation
Chi-Square test of independence


*/

WITH contingency_table AS (
  SELECT cost_category, book_table, COUNT(*) AS count
  FROM `suggester-423307.cleaned_file.table1`
  GROUP BY cost_category, book_table
),
row_totals AS (
  SELECT cost_category, SUM(count) AS row_total
  FROM contingency_table
  GROUP BY cost_category
),
col_totals AS (
  SELECT book_table, SUM(count) AS col_total
  FROM contingency_table
  GROUP BY book_table
),
total_count AS (
  SELECT SUM(count) AS total_count
  FROM contingency_table
),
expected_counts AS (
  SELECT c.cost_category, c.book_table,
    (r.row_total * co.col_total) / t.total_count AS expected_count
  FROM contingency_table c
  JOIN row_totals r ON c.cost_category = r.cost_category
  JOIN col_totals co ON c.book_table = co.book_table
  CROSS JOIN total_count t
)
SELECT SUM(POWER(count - expected_count, 2) / expected_count) AS chi_square_statistic
FROM contingency_table
JOIN expected_counts
USING (cost_category, book_table);






/*
--Question 3
Is there an association between the type of cuisine (cuisines) and the location of the restaurant (location)?

Solution: Chi-Square test of independence

Explanation
Cuisines column has multiple categories. So I decided to pick top 10 categories to perform Chi-Square test. 
Procedure I followed

1. Identify Top 10 Cuisines:
Use a query to select the top 10 categories or cuisines. 

2. Create a Temporary Table:
I'll create a temporary table to store the top 10 cuisines for easier reference.

3. Filter Main Dataset:
I'll filter the main dataset to include only restaurants serving the top 10 cuisines.

4. Normalize Cuisine Data:
Unnest the cuisines column to create a row for each cuisine.

5. Create Contingency Table:
Create a contingency table with cuisine and online_order.

6. Perform Chi-Square Test:
Calculate the Chi-Square statistic and p-value.

*/

WITH top_10_cuisines AS (
  SELECT cuisines, COUNT(*) AS cuisine_count
  FROM `suggester-423307.cleaned_file.table1`
  GROUP BY cuisines
  ORDER BY cuisine_count DESC
  LIMIT 10
),
filtered_data AS (
  SELECT name, online_order, book_table, rate, votes, location, rest_type, dish_liked, cuisines,cost,reviews_list,type,cost_category,bag_of_words, lemmatized_reviews, SPLIT(cuisines, ',') AS cuisine_array
  FROM `suggester-423307.cleaned_file.table1`
  WHERE cuisines IN (SELECT cuisines FROM top_10_cuisines)
),
unnested_data AS (
  SELECT name, online_order,book_table, rate, votes, location,rest_type, dish_liked,cuisine, cost, reviews_list, type, cost_category, bag_of_words,lemmatized_reviews
  FROM filtered_data,
    UNNEST(cuisine_array) AS cuisine
),
contingency_table AS (
  SELECT cuisine, online_order,COUNT(*) AS count
  FROM unnested_data
  GROUP BY cuisine, online_order
),
row_totals AS (
  SELECT cuisine, SUM(count) AS row_total
  FROM contingency_table
  GROUP BY cuisine
),
col_totals AS (
  SELECT online_order, SUM(count) AS col_total
  FROM contingency_table
  GROUP BY online_order
),
total_count AS (
  SELECT SUM(count) AS total_count
  FROM contingency_table
),
expected_counts AS (
  SELECT c.cuisine, c.online_order,
    (r.row_total * co.col_total) / t.total_count AS expected_count
  FROM contingency_table c
  JOIN row_totals r ON c.cuisine = r.cuisine
  JOIN col_totals co ON c.online_order = co.online_order
  CROSS JOIN total_count t
)
SELECT SUM(POWER(count - expected_count, 2) / expected_count) AS chi_square_statistic
FROM contingency_table
JOIN expected_counts
USING (cuisine, online_order);

/*
--Question 4:
Does the availability of online ordering (online_order) impact the restaurant's rating (rate)?

--Solution:
While rate is numerical, I will categorize it into ranges ( high, medium, low) and then apply Chi-Square.

--Procedure:
1. Identify Top 10 Cuisines:
-Create a query to identify the top 10 cuisines based on frequency.
-Store the results in a temporary table or a subquery for later use.

2. Filter Data:
-Filter the main dataset to include only restaurants serving the top 10 cuisines.

3. Normalize Cuisine Data:
-Split the cuisines column into individual cuisines and create a new row for each cuisine.

4. Categorize Rating:
-Create a new column to categorize the rate column into rating categories (low, medium, high).

5. Create Contingency Table:
-Create a contingency table with online_order and rating_category to count occurrences of each combination.

6. Calculate Expected Frequencies:
-Calculate the expected frequencies for each cell in the contingency table.

7. Calculate Chi-Square Statistic:
-Use the formula to calculate the Chi-Square statistic.

8. Determine Significance:
-Compare the calculated Chi-Square statistic with a critical value or calculate a p-value to determine significance.

9. Interpret Results:
-Based on the p-value, determine if there's a significant relationship between online_order and rating_category.
*/
WITH top_10_cuisines AS (
  SELECT cuisines, COUNT(*) AS cuisine_count
  FROM `suggester-423307.cleaned_file.table1`
  GROUP BY cuisines
  ORDER BY cuisine_count DESC
  LIMIT 10
),
filtered_data AS (
  SELECT 
    name,
    online_order,
    book_table,
    rate,
    votes,
    location,
    rest_type,
    dish_liked,
    cuisines,
    cost,
    reviews_list,
    type,
    cost_category,
    bag_of_words,
    lemmatized_reviews,
    SPLIT(cuisines, ',') AS cuisine_array
  FROM `suggester-423307.cleaned_file.table1`
  WHERE cuisines IN (SELECT cuisines FROM top_10_cuisines)
),
unnested_data AS (
  SELECT
    name,
    online_order,
    book_table,
    rate,
    votes,
    location,
    rest_type,
    dish_liked,
    cuisine,
    cost,
    reviews_list,
    type,
    cost_category,
    bag_of_words,
    lemmatized_reviews,
    CASE
      WHEN rate <= 2 THEN 'low_rating'
      WHEN rate > 2 AND rate < 4 THEN 'medium_rating'
      WHEN rate > 4 THEN 'high_rating'
      ELSE 'other'
    END AS rating_category
  FROM
    filtered_data,
    UNNEST(cuisine_array) AS cuisine
),
contingency_table AS (
  SELECT
    online_order,
    rating_category,
    COUNT(*) AS count
  FROM
    unnested_data
  GROUP BY
    online_order,
    rating_category
),
row_totals AS (
  SELECT rating_category, SUM(count) AS row_total
  FROM contingency_table
  GROUP BY rating_category
),
col_totals AS (
  SELECT online_order, SUM(count) AS col_total
  FROM contingency_table
  GROUP BY online_order
),
total_count AS (
  SELECT SUM(count) AS total_count
  FROM contingency_table
),
expected_counts AS (
  SELECT c.rating_category, c.online_order,
    (r.row_total * co.col_total) / t.total_count AS expected_count
  FROM contingency_table c
  JOIN row_totals r ON c.rating_category = r.rating_category
  JOIN col_totals co ON c.online_order = co.online_order
  CROSS JOIN total_count t
)
SELECT SUM(POWER(count - expected_count, 2) / expected_count) AS chi_square_statistic
FROM contingency_table
JOIN expected_counts
USING (online_order, rating_category);


--NUMERICAL COLUMNS STATISTICAL ANALYSIS

/*
--Question 1
1. Is there a significant difference in the average rating (rate) between restaurants that offer online ordering and those that don't?
I want to determine if there's a statistically significant difference in the average rating between restaurants that offer online ordering and those that don't.

Solution: Independent Samples T-test
A two-sample independent t-test is suitable for comparing the means of two independent groups (restaurants with and without online ordering) in this case.


Procedure: 

*/

WITH data_with_groups AS (
  SELECT
    *,
    CASE WHEN online_order = TRUE THEN 'with_online_order' ELSE 'without_online_order' END AS order_group
  FROM `suggester-423307.cleaned_file.table1`
)
SELECT AVG(rate) AS average_rating, order_group
FROM data_with_groups
GROUP BY order_group;


/* 
--Question 2
2. Does the cost category (cost) influence the average rating (rate) of a restaurant?

Solution: ANOVA

Procedure:
1. Calculate group means: Determine the average rating for each cost category.
2. Calculate overall mean: Find the overall average rating across all data points.
3. Calculate Sum of Squares Between Groups (SSB): Measure the variation between group means.
4. Calculate Sum of Squares Within Groups (SSW): Measure the variation within each group.
5. Calculate Mean Square Between Groups (MSB): Divide SSB by the degrees of freedom between groups.
6. Calculate Mean Square Within Groups (MSW): Divide SSW by the degrees of freedom within groups.
7. Calculate F-statistic: Calculate the ratio of MSB to MSW.
8. Determine significance: Compare the calculated F-statistic to a critical F-value or calculate a p-value.
*/

SELECT cost_category, AVG(rate) AS average_rating
FROM `suggester-423307.cleaned_file.table1`
GROUP BY cost_category;

/*
--Question 3
3. Is there a significant difference in the average number of votes (votes) between different types of restaurants (rest_type)?

Solution: ANOVA

Procedure:

*/





